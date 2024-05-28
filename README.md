# Fault Tolerant ML: Efficient Meta-Aggregation and Synchronous Training

[![ICML 2024 Paper](https://img.shields.io/badge/ICML%202024-Paper-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-2207.14287v1-B31B1B.svg)](https://arxiv.org/abs/2405.14759)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data associated with the paper **"Fault Tolerant ML: Efficient Meta-Aggregation and Synchronous Training"** by Tehila Dahan and Kfir Y. Levy. The paper investigates Byzantine-robust training in distributed machine learning systems, introducing the Centered Trimmed Meta Aggregator (CTMA) and a double-momentum strategy for gradient estimation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Setting up Weights & Biases](#setting-up-weights--biases)
  - [Running the Experiments](#running-the-experiments)
- [Results](#results)
- [License](#license)

## Introduction

This repository features the implementation of the Centered Trimmed Meta Aggregator (CTMA) and showcases the effectiveness of the double-momentum gradient estimation technique in enhancing fault tolerance. Our approach improves the efficiency and practicality of Byzantine-robust training in distributed ML systems. Detailed theoretical insights and empirical results are discussed in the paper.

## Installation

Clone the repository and navigate to the `synchronous-fault-tolerant-ml` package directory:

```bash
git clone https://github.com/dahan198/synchronous-fault-tolerant-ml.git
cd synchronous-fault-tolerant-ml
```

### Installing PyTorch

Make sure you have PyTorch installed. You can install it by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).


To install the necessary dependencies, please run:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python main.py --dataset [DATASET_NAME] --model [MODEL_NAME] --optimizer [OPTIMIZER_NAME] --iterations_num [NUM] --eval_interval [NUM] --learning_rate [LR] --batch_size [SIZE] --seed [SEED]
```

### Setting up Weights & Biases

To use Weights & Biases for visualizing training results, you need to install `wandb` and log in to your account:

```bash
pip install wandb
wandb login
```

After logging in, create a `wandb.yaml` configuration file in the `./config` directory with your project name and entity:

```yaml
# ./config/wandb.yaml
project: your_project_name
entity: your_entity_name
```

Replace `your_project_name` and `your_entity_name` with your actual wandb project and entity names.


Enable visualization with Weights & Biases:

```bash
python main.py --use_wandb ...
```

### Running the Experiments

Choose the configuration you want to run and execute the corresponding command:

```bash
python run_confX.py
```

Replace `run_confX.py` with the appropriate script for your desired experiment, where `X` is the number of the configuration you wish to run (e.g., `run_conf1.py`, `run_conf2.py`, etc.).
## Results

The results of our experiments are summarized in the paper. We evaluate the performance of CTMA and the double-momentum technique across various models and datasets, demonstrating significant improvements in robustness and efficiency compared to baseline methods.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
