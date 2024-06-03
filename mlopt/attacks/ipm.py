import torch
from .attack import Attack


class IPMAttack(Attack):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def apply(self, inputs, targets, honest_updates, worker, gradient_function):
        return - self.epsilon * torch.mean(honest_updates, dim=0)