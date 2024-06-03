import torch
import random
import math


class Bucketing:
    """
        This class is an adaptation of the bucketing as originally implemented in 'bucketing.py' by John Stephan,
        from the paper "Fixing by mixing: A recipe for optimal byzantine ML under heterogeneity"
        by Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafaël Pinot, and John Stephan,
        presented at the International Conference on Artificial Intelligence and Statistics 2023 (pages 1232-1300).

        Copyright © 2022 École Polytechnique Fédérale de Lausanne (EPFL). All rights reserved.

        The original implementation was part of a project described in the paper
        "Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing" by Sai Praneeth Karimireddy,
        Lie He, and Martin Jaggi, presented at ICLR 2022.
    """

    def __init__(self, aggregator, num_workers, num_byzantine_workers, **kwargs):
        self.aggregator = aggregator
        self.bucket_size = math.floor(num_workers / (2 * num_byzantine_workers))
        if num_byzantine_workers > num_workers / 4:
            raise ValueError("Too many byzantine workers for bucketing")
        self.num_workers = num_workers
        self.worker_ids = list(range(num_workers))

    def __call__(self, gradients, weights=None):
        processed_gradients = []
        random.shuffle(self.worker_ids)
        shuffled_gradients = gradients[self.worker_ids]
        num_buckets = math.ceil(self.num_workers / self.bucket_size)
        buckets = [shuffled_gradients[i:i + self.bucket_size] for i in range(0, self.num_workers, self.bucket_size)]

        for bucket in range(num_buckets):
            processed_gradients.append(buckets[bucket].mean(dim=0))
        return self.aggregator(torch.stack(processed_gradients))


