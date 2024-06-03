"""
@misc{karimireddy2021byzantinerobust,
      title={Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing},
      author={Sai Praneeth Karimireddy and Lie He and Martin Jaggi},
      year={2021},
      eprint={2006.09365},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

Code adapted from: https://github.com/epfml/byzantine-robust-noniid-optimizer/tree/main

MIT License

Copyright (c) 2021 EPFL Machine Learning and Optimization Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import numpy as np
from scipy.stats import norm
from .attack import Attack


class ALittleIsEnoughAttack(Attack):

    def __init__(self, workers_num, byzantine_num):
        s = np.floor(workers_num / 2 + 1) - byzantine_num
        cdf_value = (workers_num - byzantine_num - s) / (workers_num - byzantine_num)
        self.z_max = norm.ppf(cdf_value)

    def apply(self, inputs, targets, honest_updates, worker, gradient_function):
        mu = torch.mean(honest_updates, dim=0)
        std = torch.std(honest_updates, dim=0)
        return mu - std * self.z_max
