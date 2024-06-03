from  .nnm import NNM
from .centered_trimmed import CenteredTrimmedMetaAggregator


class NNMixingAndCTMetaAggregator:
    def __init__(self, agg2boost, num_workers, num_byzantine):
        self.nnm = NNM(agg2boost, num_byzantine)
        self.ctma = CenteredTrimmedMetaAggregator(self.nnm, num_workers, num_byzantine)
        self.aggregator = agg2boost
        self.num_honest = num_workers - num_byzantine

    def __call__(self, gradients, weights=None):
        return self.ctma(gradients)
