from .attack import Attack


class SignFlippingAttack(Attack):

    def __init__(self, device):
        self.device = device

    def apply(self, inputs, targets, honest_updates, worker, gradient_function):
        gradient, __, __ = gradient_function(inputs, targets)
        return - worker.step(gradient)

