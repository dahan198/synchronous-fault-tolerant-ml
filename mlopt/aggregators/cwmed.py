import torch


class CWMed:
    def __init__(self, **kwargs):
        pass

    def __call__(self, gradients, weights=None):
        return torch.median(gradients, dim=0)[0], None

