# implementation from https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/convolutional.py


import torch
from torch import nn
import inspect

from src.modules.models import norms, activations



def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if ``fn`` can be called with ``name`` as a keyword
            argument.

    Returns:
        bool: whether ``fn`` accepts a ``name`` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )

class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm.
            -  ``'cLN'``: channelwise Layernorm.
            -  ``'cgLN'``: cumulative global Layernorm.
            -  Any norm supported by :func:`~.norms.get`
        causal (bool, optional) : Whether or not the convolutions are causal


    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        causal (bool, optional) : Whether or not the convolutions are causal.

    References
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2**x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2**x
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection += skip
            else:
                residual = tcn_out
            output += residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "causal": self.causal,
        }
        return config


class WaveformEncoder(nn.Module):
    def __init__(self, num_filters, kernel_size, stride=None):
        super(WaveformEncoder, self).__init__()
        stride = stride if stride else kernel_size // 2

        self.conv1d = nn.Conv1d(
            in_channels=1,  # Assuming single-channel (mono) input
            out_channels=num_filters,  # Number of basis filters
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size - stride) + 1) // 2,  # Padding to maintain the resolution
            bias=False
        )

    def forward(self, x):
        # Forward pass: apply the learnable Conv1D
        basis_coefficients = self.conv1d(x)  # Shape: (batch_size, num_filters, num_frames)
        return basis_coefficients


class WaveformDecoder(nn.Module):
    def __init__(self, num_filters, kernel_size, stride=None):
        super(WaveformDecoder, self).__init__()
        stride = stride if stride else kernel_size // 2

        self.deconv1d = nn.ConvTranspose1d(
            in_channels=num_filters,
            out_channels=1,  # Output is a single-channel waveform
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size - stride) + 1) // 2,  # Padding similar to encoder
            bias=False
        )

    def forward(self, masked_basis):
        if masked_basis.ndim < 4:
            return self.deconv1d(masked_basis)

        separated_sources = []
        # Decode each source using the transposed convolution
        for i in range(masked_basis.size(1)):
            separated = self.deconv1d(masked_basis[:, i])  # Decode each source separately
            separated_sources.append(separated)

        # Stack all separated sources
        separated_sources = torch.stack(separated_sources, dim=1)  # Shape: (batch_size, num_sources, 1, num_samples)

        return separated_sources.squeeze(2)


class TDConvNetpp(nn.Module):
    """Improved Temporal Convolutional network used in [1] (TDCN++)

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.

    References
        [1] : Kavalerov, Ilya et al. “Universal Sound Separation.” in WASPAA 2019

    .. note::
        The differences wrt to ConvTasnet's TCN are:

        1. Channel wise layer norm instead of global
        2. Longer-range skip-residual connections from earlier repeat inputs
           to later repeat inputs after passing them through dense layer.
        3. Learnable scaling parameter after each dense layer. The scaling
           parameter for the second dense  layer  in  each  convolutional
           block (which  is  applied  rightbefore the residual connection) is
           initialized to an exponentially decaying scalar equal to 0.9**L,
           where L is the layer or block index.

    """

    def __init__(
        self,
        in_chan,
        n_src,
        encoder,
        decoder,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="fgLN",
        mask_act="relu",
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.encoder = encoder
        self.decoder = decoder

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2**x // 2
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                    )
                )
        # Dense connection in TDCNpp
        self.dense_skip = nn.ModuleList()
        for r in range(n_repeats - 1):
            self.dense_skip.append(nn.Conv1d(bn_chan, bn_chan, 1))

        scaling_param = torch.Tensor([0.9**l for l in range(1, n_blocks)])
        scaling_param = scaling_param.unsqueeze(0).expand(n_repeats, n_blocks - 1).clone()
        self.scaling_param = nn.Parameter(scaling_param, requires_grad=True)

        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        out_size = skip_chan if skip_chan else bn_chan
        self.consistency = nn.Linear(out_size, n_src)

    def forward(self, wave):
        r"""Forward.

        Args:
            wave (:class:`torch.Tensor`): Tensor of shape $(batch, time)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """

        wave = wave.unsqueeze(1)
        encoded_wave = self.encoder(wave)

        batch, n_filters, n_frames = encoded_wave.size()
        output = self.bottleneck(encoded_wave)
        output_copy = output

        skip_connection = 0.0
        for r in range(self.n_repeats):
            # Long range skip connection TDCNpp
            if r != 0:
                # Transform the input to repeat r-1 and add to new repeat inp
                output = self.dense_skip[r - 1](output_copy) + output
                # Copy this for later.
                output_copy = output
            for x in range(self.n_blocks):
                # Common to w. skip and w.o skip architectures
                i = r * self.n_blocks + x
                tcn_out = self.TCN[i](output)
                if self.skip_chan:
                    residual, skip = tcn_out
                    skip_connection = skip_connection + skip
                else:
                    residual = tcn_out
                # Initialized exp decay scale factor TDCNpp for residual connections
                scale = self.scaling_param[r, x - 1] if x > 0 else 1.0
                residual = residual * scale
                output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)

        weights = self.consistency(mask_inp.mean(-1))
        weights = torch.nn.functional.softmax(weights, -1)
        masked_tf_rep = est_mask * encoded_wave.unsqueeze(1)

        est_source = self.decoder(masked_tf_rep)
        est_source = mixture_consistency(wave, est_source, weights.unsqueeze(-1))

        return est_mask, est_source, weights

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
        }
        return config


from typing import Optional, List


def mixture_consistency(
    mixture: torch.Tensor,
    est_sources: torch.Tensor,
    src_weights: Optional[torch.Tensor] = None,
    dim: int = 1,
) -> torch.Tensor:
    """Applies mixture consistency to a tensor of estimated sources.

    Args:
        mixture (torch.Tensor): Mixture waveform or TF representation.
        est_sources (torch.Tensor): Estimated sources waveforms or TF representations.
        src_weights (torch.Tensor): Consistency weight for each source.
            Shape needs to be broadcastable to `est_source`.
            We make sure that the weights sum up to 1 along dim `dim`.
            If `src_weights` is None, compute them based on relative power.
        dim (int): Axis which contains the sources in `est_sources`.

    Returns
        torch.Tensor with same shape as `est_sources`, after applying mixture
        consistency.

    Examples
        >>> # Works on waveforms
        >>> mix = torch.randn(10, 16000)
        >>> est_sources = torch.randn(10, 2, 16000)
        >>> new_est_sources = mixture_consistency(mix, est_sources, dim=1)
        >>> # Also works on spectrograms
        >>> mix = torch.randn(10, 514, 400)
        >>> est_sources = torch.randn(10, 2, 514, 400)
        >>> new_est_sources = mixture_consistency(mix, est_sources, dim=1)

    .. note::
        This method can be used only in 'complete' separation tasks, otherwise
        the residual error will contain unwanted sources. For example, this
        won't work with the task `"sep_noisy"` from WHAM.

    References
        Scott Wisdom et al. "Differentiable consistency constraints for improved
        deep speech enhancement", ICASSP 2019.
    """
    # If the source weights are not specified, the weights are the relative
    # power of each source to the sum. w_i = P_i / (P_all), P for power.
    if src_weights is None:
        all_dims: List[int] = torch.arange(est_sources.ndim).tolist()
        all_dims.pop(dim)  # Remove source axis
        all_dims.pop(0)  # Remove batch axis
        src_weights = torch.mean(est_sources**2, dim=all_dims, keepdim=True)
    # Make sure that the weights sum up to 1
    norm_weights = torch.sum(src_weights, dim=dim, keepdim=True) + 1e-8
    src_weights = src_weights / norm_weights

    # Compute residual mix - sum(est_sources)
    if mixture.ndim == est_sources.ndim - 1:
        # mixture (batch, *), est_sources (batch, n_src, *)
        residual = (mixture - est_sources.sum(dim=dim)).unsqueeze(dim)
    elif mixture.ndim == est_sources.ndim:
        # mixture (batch, 1, *), est_sources (batch, n_src, *)
        residual = mixture - est_sources.sum(dim=dim, keepdim=True)
    else:
        n, m = est_sources.ndim, mixture.ndim
        raise RuntimeError(
            f"The size of the mixture tensor should match the "
            f"size of the est_sources tensor. Expected mixture"
            f"tensor to have {n} or {n-1} dimension, found {m}."
        )
    # Compute remove
    new_sources = est_sources + src_weights * residual
    return new_sources


