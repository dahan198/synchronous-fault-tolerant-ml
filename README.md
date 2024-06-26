# Fault Tolerant ML: Efficient Meta-Aggregation and Synchronous Training

[![ICML 2024 Paper](https://img.shields.io/badge/ICML%202024-Paper-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-2207.14287v1-B31B1B.svg)](https://arxiv.org/abs/2405.14759)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data associated with the paper **"Fault Tolerant ML: Efficient Meta-Aggregation and Synchronous Training"** by Tehila Dahan and Kfir Y. Levy. The paper investigates Byzantine-robust training in distributed machine learning systems, introducing the Centered Trimmed Meta Aggregator (CTMA) and showcasing the effectiveness of the double-momentum gradient estimation technique in enhancing fault tolerance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Setting up Weights & Biases](#setting-up-weights--biases)
- [License](#license)

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
