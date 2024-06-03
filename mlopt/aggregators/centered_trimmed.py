import torch


class CenteredTrimmedMetaAggregator:
    def __init__(self, agg2boost, num_workers, num_byzantine):
        self.aggregator = agg2boost
        self.num_honest = num_workers - num_byzantine

    def trim(self, v):
        v_norm = torch.norm(v, dim=1)
        scale = v_norm <= torch.kthvalue(v_norm, self.num_honest).values
        byzantine_indices = torch.nonzero(~scale).squeeze()
        return scale.unsqueeze(1) * v, byzantine_indices

    def __call__(self, gradients, weights=None):
        aggregated, _ = self.aggregator(gradients)
        trimmed, byz_idx = self.trim(gradients - aggregated)
        aggregated += torch.sum(trimmed, dim=0) / self.num_honest
        return aggregated, byz_idx


