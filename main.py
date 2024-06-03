import argparse
from mlopt.dataset import DATASET_REGISTRY
from mlopt.model import MODEL_REGISTRY
from mlopt.trainer.sync_trainer import SyncTrainer
from mlopt.optimizer import OPTIMIZER_REGISTRY
from torch.utils.data import DataLoader
from mlopt.utils import set_seed, get_device
from mlopt.aggregators import get_aggregator
from mlopt.worker import *
from mlopt.attacks import get_attack


def main():
    parser = argparse.ArgumentParser(description="Training script for synchronous Byzantine machine learning.")

    # Define the arguments
    parser.add_argument('--boost_agg', type=str, default=None,
                        choices=[None, 'ctma', 'nnm', 'nnm+ctma', 'bucketing'],
                        help='Boost aggregation method to be used. '
                             'Choices include: ctma, nnm, nnm+ctma, and bucketing.')
    parser.add_argument('--agg', type=str, default='avg', choices=['avg', 'cwmed', 'cwtm', 'rfa'],
                        help='Aggregation method for combining gradients. '
                             'Choices include: avg, cwmed, cwtm, and rfa.')
    parser.add_argument('--attack', type=str, default='lf', choices=[None, 'lf', 'sf', 'empire', 'little'],
                        help='Type of attack to be simulated during training. '
                             'Choices include: lf, sf, empire, and little.')
    parser.add_argument('--workers_num', type=int, default=17,
                        help='Number of workers to be used for training.')
    parser.add_argument('--byzantine_num', type=int, default=8,
                        help='Number of Byzantine workers to be simulated.')
    parser.add_argument('--config_folder_path', type=str, default='./config',
                        help='Path to the configuration folder.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to be used for training. Choices include: mnist and cifar10.')
    parser.add_argument('--model', type=str, default='simple_mnist', choices=MODEL_REGISTRY.keys(),
                        help='Model architecture to be used.')
    parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--eval_interval', type=int, default=20,
                        help='Interval (in epochs) at which evaluation is performed.')
    parser.add_argument('--optimizer', type=str, default='mu2sgd', choices=['mu2sgd', 'momentum'],
                        help='Optimizer to be used for training.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--gradient_momentum', type=float, default=0.9,
                        help='Fixed momentum parameter for the gradient for the mometum optimizer '
                             'and for the mu2sgd if use_beta_t is False..')
    parser.add_argument('--use_alpha_t', action='store_true', default=True,
                        help='Flag to enable the use of alpha_t=t feature in the mu2sgd optimizer.')
    parser.add_argument('--use_beta_t', action='store_true', default=True,
                        help='Flag to enable the use of beta_t=1/t feature in the mu2sgd optimizer.')
    parser.add_argument('--query_point_momentum', type=float, default=0.1,
                        help='Fixed momentum for the query point for the mu2sgd optimizer if use_alpha_t is False.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    parser.add_argument('--store_results', action='store_true', help='Flag to store training results.')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Flag to enable visualization of results with Weights & Biases.')
    parser.add_argument('--ipm_epsilon', type=float, default=0.5,
                        help='Epsilon value for the IPM attack (if used).')
    parser.add_argument('--gradient_clip', type=float, default=2.0,
                        help='Value for gradient clipping. If not used, set to None.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay value for the optimizer.')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of the experiment for logging and identification.')

    # Parse the arguments
    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device()
    attack = None
    if args.attack:
        attack = get_attack(args.attack, device=device, epsilon=args.ipm_epsilon, workers_num=args.workers_num,
                            byzantine_num=args.byzantine_num)
    if args.boost_agg is not None:
        agg2boost = get_aggregator(args.agg, args.workers_num, args.byzantine_num)
        aggregator = get_aggregator(args.boost_agg, args.workers_num, args.byzantine_num,
                                    agg2boost=agg2boost)
    else:
        aggregator = get_aggregator(args.agg, args.workers_num, args.byzantine_num)
    model = MODEL_REGISTRY[args.model]().to(device).double()

    dataset = DATASET_REGISTRY[args.dataset]()
    train_dataloader = DataLoader(dataset.trainset, batch_size=args.batch_size * args.workers_num, shuffle=True)
    test_dataloader = DataLoader(dataset.testset, batch_size=args.batch_size * args.workers_num, shuffle=False)
    if args.optimizer == "momentum":
        optimizer = OPTIMIZER_REGISTRY["sgd"]
        workers = [WorkerMomentum(beta=args.gradient_momentum) for _ in range(args.workers_num)]
        optimizer_params = {"lr": args.learning_rate,
                            "momentum": 0.0,
                            "weight_decay": args.weight_decay}
    else:
        optimizer = OPTIMIZER_REGISTRY["anytime_sgd"]
        workers = [WorkerSTORM(beta=args.gradient_momentum, use_beta_t=args.use_beta_t, ) for _ in
                   range(args.workers_num)]
        optimizer_params = {"lr": args.learning_rate,
                            "gamma": args.query_point_momentum,
                            "use_alpha_t": args.use_alpha_t,
                            "weight_decay": args.weight_decay}
    optimizer = optimizer(model.parameters(), **optimizer_params)
    trainer = SyncTrainer(model, optimizer, train_dataloader, test_dataloader, args, aggregator,
                          workers, args.byzantine_num, attack, args.gradient_clip,
                          args.experiment_name)
    trainer.train(epoch_num=args.epoch_num, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()
