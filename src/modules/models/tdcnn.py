import torch.nn as nn


class mixit(nn.Module):
    def __init__(self, cfg, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError()

