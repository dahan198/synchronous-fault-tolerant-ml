import torch


class RobustFederatedAveraging:

    def __init__(self, T=3, nu=0., **kwargs):
        self.T = T
        self.nu = nu

    def __call__(self, gradients, weights=None):
        alphas = [1 / len(gradients) for _ in range(len(gradients))]
        z = torch.zeros_like(gradients[0])
        return smoothed_weiszfeld(gradients, alphas, z=z, nu=self.nu, T=self.T), None


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def smoothed_weiszfeld(vector, alphas, z, nu, T):
    m = len(vector)
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    for t in range(T):
        betas = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, vector[k])
            betas.append(alphas[k] / max(distance, nu))

        z = 0
        for w, beta in zip(vector, betas):
            z += w * beta
        z /= sum(betas)
    return z


