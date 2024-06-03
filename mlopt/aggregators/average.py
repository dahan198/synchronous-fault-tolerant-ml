import torch


class Average:
    def __init__(self, **kwargs):
        pass

    def __call__(self, gradients, weights=None):
        return torch.mean(gradients, dim=0), None

