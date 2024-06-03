from torch.optim import Optimizer
import copy


class AnyTimeSGD(Optimizer):
    def __init__(self, params, lr=0.01, gamma=0.9, use_alpha_t=False, weight_decay=0.0,
                 min_gamma=None):
        """
        Initializes the custom SGD optimizer with weight decay.

        Args:
        - params (iterable): An iterable of `torch.Tensor`s or `dict`s. Specifies what Tensors should be optimized.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay coefficient (L2 penalty).
        """
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AnyTimeSGD, self).__init__(params, defaults)
        self.gamma = gamma
        self.iter = 0
        self.sum_iter = 0
        self.use_alpha_t = use_alpha_t
        self.min_gamma = min_gamma
        self.w = []
        self.w_s = []
        self.x_s = []
        for group in self.param_groups:
            cloned_group = copy.deepcopy(group)
            for i, p in enumerate(group['params']):
                cloned_group['params'][i] = p.clone().detach().requires_grad_(p.requires_grad)

            self.w.append(cloned_group)

    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).

        Args:
        - closure (callable, optional): A closure that reevaluates the model and returns the loss. Optional for most optimizers.
        """
        loss = None
        self.iter += 1
        self.sum_iter += self.iter
        if closure is not None:
            loss = closure()

        for group, group_w in zip(self.param_groups, self.w):
            weight_decay = group['weight_decay']
            for i, (p, pw) in enumerate(zip(group['params'], group_w['params'])):
                if p.grad is None:
                    continue
                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                lr = group['lr']

                # Apply update
                pw.data.add_(grad, alpha=-lr)

                if self.use_alpha_t:
                    gamma = self.iter / self.sum_iter
                    gamma = max([round(gamma, 3), 0.01])
                    if self.min_gamma is not None:
                        gamma = max(gamma, self.min_gamma)
                    p.data.mul_(1 - gamma).add_(pw.data, alpha=gamma)
                else:
                    p.data.mul_(1 - self.gamma).add_(pw.data, alpha=self.gamma)
        return loss