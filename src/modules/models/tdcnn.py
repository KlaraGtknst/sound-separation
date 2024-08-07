import torch.nn as nn


class TDCNN(nn.Module):
    def __init__(self, cfg, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError()

