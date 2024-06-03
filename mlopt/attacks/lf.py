from .attack import Attack


class LabelFlippingAttack(Attack):

    def __init__(self, device):
        self.device = device

    def apply(self, inputs, targets, honest_updates, worker, gradient_function):
        gradient, __, __ = gradient_function(inputs, 9 - targets)
        return worker.step(gradient)
