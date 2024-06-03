import torch


class CWTrimmedMean:

    def __init__(self, num_byzantine, **kwargs):
        self.num_byzantine = num_byzantine

    def __call__(self, gradients, weights=None):
        if len(gradients) - 2 * self.num_byzantine <= 0:
            raise ValueError("Too many elements to trim; not enough gradients left to compute mean.")

        # Sort the gradients for each dimension
        sorted_gradients, _ = torch.sort(gradients, dim=0)

        # Trim the top and bottom b gradients
        trimmed_gradients = sorted_gradients[self.num_byzantine:-self.num_byzantine] if self.num_byzantine > 0 else sorted_gradients

        # Compute the mean of the remaining gradients
        mean_gradients = trimmed_gradients.mean(dim=0)
        return mean_gradients, None


