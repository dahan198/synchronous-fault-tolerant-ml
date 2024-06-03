import torch
import os
from .trainer import Trainer
from prettytable import PrettyTable
from tqdm import tqdm
import yaml


class SyncTrainer(Trainer):

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, params, aggregator, workers,
                 byzantine_num, attack, gradient_clip=None, experiment_name=None):
        super().__init__(model, optimizer, train_dataloader, test_dataloader, params, aggregator, workers,
                         byzantine_num, attack, gradient_clip, experiment_name)
        if self.params["use_wandb"]:
            with open(os.path.join(params.config_folder_path, "wandb.yaml"), 'r') as file:
                self.wandb_conf = yaml.safe_load(file)
            algorithm = "Mu2SGD" if self.params['optimizer'] == "mu2sgd" else "Momentum"
            project = self.wandb_conf["project"] if experiment_name is None else experiment_name
            self.wandb.init(project=project, entity=self.wandb_conf["entity"],
                            name=f"{algorithm}--{self.params['dataset']}--{self.params['model']}--{self.params['attack']}-"
                                 f"--{self.params['agg']}+{self.params['boost_agg']}--LR: {self.params['learning_rate']}--"
                                 f"--Seed: {self.params['seed']}--Batch: {self.params['batch_size']}"
                                 f"--Workers: {self.params['workers_num']}--Byz: {self.params['byzantine_num']}")
            self.wandb.config.update(self.params)

    def train(self, epoch_num: int = 100, eval_interval: int = None):
        super().train()
        self.model.train()  # Set the model to training mode
        self.accuracy_metric.reset()

        (data, label) = next(iter(self.train_dataloader))
        batch_size = data.shape[0]
        self.make_optimization_step(data.double(), label, first_step=True)

        # Initialize the metrics table outside the loop if not already done
        metrics_table = PrettyTable()
        iter_title = "Epoch" if eval_interval is None else "Iteration"
        metrics_table.field_names = [iter_title, "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]

        if eval_interval is not None:
            total_iterations = int(epoch_num * (len(self.train_dataloader) / eval_interval))
        else:
            total_iterations = epoch_num

        running_loss = 0.0
        k = 0

        for epoch in range(epoch_num):

            if eval_interval is not None:
                for inputs, labels in tqdm(self.train_dataloader):
                    if inputs.shape[0] < batch_size:
                        continue
                    k += 1
                    outputs, loss = self.make_optimization_step(inputs.double(), labels, False)
                    running_loss += loss
                    predictions = torch.argmax(outputs, dim=1)
                    self.accuracy_metric.update(predictions, labels[:predictions.shape[0]].to(self.device))

                    if k % eval_interval == 0:
                        self.make_evaluation_step((k // eval_interval) - 1, total_iterations, eval_interval,
                                                  running_loss,
                                                  metrics_table,
                                                  "Iteration")
                        running_loss = 0.0
            else:
                running_loss = 0.0
                for inputs, labels in tqdm(self.train_dataloader):
                    if inputs.shape[0] < batch_size:
                        continue
                    outputs, loss = self.make_optimization_step(inputs.double(), labels, False)
                    running_loss += loss
                    predictions = torch.argmax(outputs, dim=1)
                    self.accuracy_metric.update(predictions, labels[:predictions.shape[0]].to(self.device))
                self.make_evaluation_step(epoch, epoch_num, len(self.train_dataloader), running_loss, metrics_table,
                                          "Epoch")

        self.save_metrics_and_params()

        print('Finished Training')

    def make_optimization_step(self, inputs, targets, first_step=False, make_opt_step=True):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        workers_inputs, workers_labels = self.get_workers_samples(inputs, targets)
        outputs = None
        loss = None

        # STORM
        if self.workers[0].two_passes:
            if first_step:
                self.compute_estimator(workers_inputs, workers_labels)
            else:
                loss, outputs, workers_gradients = self.workers_optimization_step(workers_inputs, workers_labels)
                self.optimizer.step()

                # Make the second pass
                self.compute_estimator(workers_inputs, workers_labels)
        # MOMENTUM
        else:
            if first_step:
                pass
            else:
                loss, outputs, workers_gradients = self.workers_optimization_step(workers_inputs, workers_labels)
                self.optimizer.step()

        return outputs, loss

    def get_workers_samples(self, inputs, targets):
        split_num = len(self.workers)
        workers_inputs = torch.chunk(inputs, split_num, dim=0)
        workers_labels = torch.chunk(targets, split_num, dim=0)
        return workers_inputs, workers_labels

    def compute_estimator(self, workers_inputs, workers_labels):
        update_num = self.workers_num if ((self.attack.__class__.__name__ == "LabelFlippingAttack") or
                                          (self.attack.__class__.__name__ == "SignFlippingAttack")) \
            else self.honest_num
        for worker in range(update_num):
            if self.attack.__class__.__name__ == "LabelFlippingAttack" and worker >= self.honest_num:
                gradient, __, __ = self.get_worker_gradient(workers_inputs[worker], 9 - workers_labels[worker])
            else:
                gradient, __, __ = self.get_worker_gradient(workers_inputs[worker], workers_labels[worker])
            self.workers[worker].compute_estimator(gradient)

    def workers_optimization_step(self, workers_inputs, workers_labels):

        honest_momentums, loss, outputs = self.get_honest_momentums(workers_inputs, workers_labels)
        if self.attack is not None:
            byzantine_momentums = self.get_byzantine_momentums(workers_inputs, workers_labels, honest_momentums)
        else:
            byzantine_momentums = []
        workers_momentum = torch.stack(honest_momentums + byzantine_momentums)
        aggregated_momentum = self.aggregator(workers_momentum)[0]

        # Split the stacked momentums tensor into chunks according to the gradient shapes
        split_momentum = torch.split(aggregated_momentum, self.split_sizes)

        # Iterate over the model's parameters and the split momentums
        # Update each parameter's gradient with the corresponding momentum
        i = 0  # Index for the split momentums
        for p in self.model.parameters():
            momentum_reshaped = split_momentum[i].view(self.param_shapes[i]).clone()
            p.grad = momentum_reshaped
            i += 1

        return loss, outputs, workers_momentum

    def get_byzantine_momentums(self, split_inputs, split_labels, honset_momentums):
        byzantine_momentums = []
        honest_updates = torch.stack(honset_momentums)
        for worker in range(self.honest_num, self.workers_num):
            byzantine_momentums.append(self.attack.apply(split_inputs[worker],
                                                         split_labels[worker],
                                                         honest_updates,
                                                         self.workers[worker],
                                                         self.get_worker_gradient))
        return byzantine_momentums

    def get_honest_momentums(self, split_inputs, split_labels):
        workers_momentum = []
        outputs = []
        losses = 0.0
        for worker in range(self.honest_num):
            gradient, loss, output = self.get_worker_gradient(split_inputs[worker], split_labels[worker])
            losses += loss
            outputs.append(output)

            # Aggregate gradients from all batches of this loader (worker)
            workers_momentum.append(self.workers[worker].step(gradient))

        return workers_momentum, losses / self.honest_num, torch.vstack(outputs)
