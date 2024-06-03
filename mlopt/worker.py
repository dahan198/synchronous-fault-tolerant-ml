import torch


class Worker:
    def __init__(self, beta):
        self.beta = beta
        self.momentum = None
        self.two_passes = False

    def step(self, gradients):
        """

        Args:
            gradients:

        Returns:

        """


class WorkerMomentum(Worker):

    def __init__(self, beta):
        super().__init__(beta)

    def step(self, gradients):
        if self.momentum is None:
            self.momentum = gradients
        else:
            self.momentum.mul_(self.beta)
            self.momentum.add_(gradients, alpha=1 - self.beta)
        return self.momentum


class WorkerSTORM(Worker):
    def __init__(self, beta, use_beta_t):
        super().__init__(beta)
        self.g_tilde = None
        self.two_passes = True
        self.t = 1
        self.use_beta_t = use_beta_t

    def step(self, gradients):
        self.g_tilde = gradients
        return self.momentum

    def compute_estimator(self, gradients):
        if self.use_beta_t:
            self.beta = 1 / self.t
            self.t += 1
            self.beta = max([round(self.beta, 3), 0.01])
        if self.momentum is None:
            self.momentum = gradients
        else:
            difference = self.momentum.sub(self.g_tilde)
            difference.mul_(1 - self.beta)
            self.momentum.copy_(gradients)
            self.momentum.add_(difference)

