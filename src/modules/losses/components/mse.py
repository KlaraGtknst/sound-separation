from torch.nn.modules.loss import _Loss


class SingleSrcMSE(_Loss):
    r"""Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        - est_targets: :math:`(batch, ...)`.
        - targets: :math:`(batch, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)`
    """

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(
                f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
            )
        loss = (targets - est_targets) ** 2
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)

# aliases
class MultiSrcMSE(SingleSrcMSE):
    pass
