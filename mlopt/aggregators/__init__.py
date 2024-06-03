from .average import Average
from .cwmed import CWMed
from .cw_trimmed_mean import CWTrimmedMean
from .rfa import RobustFederatedAveraging
from .centered_trimmed import CenteredTrimmedMetaAggregator
from .nnm import NNM
from .bucketing import Bucketing
from .nnm_and_ctma import NNMixingAndCTMetaAggregator


aggregators_params = {
    "rfa": {
        "T": 3,
        "nu": 0.1
    }
}

AGGREGATOR_REGISTRY = {
    'avg': Average,
    'cwmed': CWMed,
    'cwtm': CWTrimmedMean,
    'rfa': RobustFederatedAveraging,
    'ctma': CenteredTrimmedMetaAggregator,
    'nnm': NNM,
    'bucketing': Bucketing,
    'nnm+ctma': NNMixingAndCTMetaAggregator,
}


def get_aggregator(aggregation, num_workers, num_byzantine, agg2boost=None):
    assert aggregation in AGGREGATOR_REGISTRY, "{} is unknown/unsupported".format(aggregation)

    # Common parameters
    common_params = {"num_workers": num_workers, "num_byzantine": num_byzantine, "agg2boost": agg2boost}

    # Load specific parameters for the aggregator from the config
    specific_params = aggregators_params.get(aggregation, {})

    # Merge common and specific parameters
    all_params = {**common_params, **specific_params}

    # Instantiate and return the aggregator
    return AGGREGATOR_REGISTRY[aggregation](**all_params)

