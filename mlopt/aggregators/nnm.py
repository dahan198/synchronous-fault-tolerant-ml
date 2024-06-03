import torch


class NNM:
    """
        This class is an adaptation of the nnm as originally implemented by John Stephan,
        from the paper "Fixing by mixing: A recipe for optimal byzantine ML under heterogeneity"
        by Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafaël Pinot, and John Stephan,
        presented at the International Conference on Artificial Intelligence and Statistics 2023 (pages 1232-1300).

        Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL). All rights reserved.
    """
    def __init__(self, agg2boost, num_byzantine, **kwargs):
        self.aggregator = agg2boost
        self.num_byzantine = num_byzantine

    def __call__(self, gradients, weights=None):
        new_gradients = list()
        for grad in gradients:
            new_gradients.append(self.compute_cva(gradients, self.num_byzantine, grad))
        return self.aggregator(torch.vstack(new_gradients))

    @staticmethod
    def compute_cva(gradients, num_byzantine, pivot_gradient):
        gradient_scores = list()
        for i in range(len(gradients)):
            distance = gradients[i].sub(pivot_gradient).norm().item()
            gradient_scores.append((i, distance))
        gradient_scores.sort(key=lambda x: x[1])
        closest_gradients = [gradients[gradient_scores[j][0]] for j in range(len(gradients) - num_byzantine)]
        stacked_grads = torch.stack(closest_gradients)
        return stacked_grads.mean(dim=0)

