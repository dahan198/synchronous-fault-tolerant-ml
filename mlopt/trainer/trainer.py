import torch.nn as nn
import torchmetrics
from ..utils import get_device
import os
import json
import csv
import wandb
import torch


class Trainer:

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, params, aggregator, workers,
                 byzantine_num, attack, gradient_clip=None, experiment_name=None):
        self.workers_num = len(workers)
        self.byzantine_num = byzantine_num
        self.honest_num = self.workers_num - self.byzantine_num
        self.attack = attack
        self.model = model
        self.criterion = None
        self.experiment_name = experiment_name
        self.gradient_clip = gradient_clip
        self.aggregator = aggregator
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.device = get_device()
        self.model.to(self.device)
        self.param_shapes = [p.shape for p in model.parameters()]
        self.split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in self.param_shapes]
        self.checkpoint_path = None
        self.log_file_path = None
        self.workers = workers
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

        self.params = params.__dict__
        self.metrics_data = []
        self.run_directory = "./"
        self.use_wandb = self.params["use_wandb"]
        self.wandb = wandb
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
        self.best_accuracy = 0.0
        self.iter_results = {"iteration": 0,
                             "train_loss": 0,
                             "train_acc": 0,
                             "test_loss": 0,
                             "test_acc": 0}
        self.iter_results = {**self.iter_results, **params.__dict__}

    def train(self):

        # Check if the results directory exists, if not, create it
        results_dir = 'results' if self.experiment_name is None else 'results-' + self.experiment_name
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Get all the existing folders in the results directory
        existing_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

        # Find the highest numbered folder
        highest_num = len(existing_folders)

        # Create a new folder with the next number
        next_folder_num = highest_num + 1
        run_directory = os.path.join(results_dir, 'run' + str(next_folder_num))
        os.makedirs(run_directory)
        self.checkpoint_path = os.path.join(results_dir, 'run' + str(next_folder_num), "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.log_file_path = os.path.join(run_directory, 'metrics_log.txt')
        self.run_directory = run_directory

        params_file_path = os.path.join(run_directory, 'params.json')

        # Save the params_dict to a JSON file
        with open(params_file_path, 'w') as params_file:
            json.dump(self.params, params_file, indent=4)

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        self.accuracy_metric.reset()  # Reset the accuracy metric

        total_loss = 0.0

        with torch.no_grad():  # No gradient is needed for evaluation
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs.double())
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                self.accuracy_metric.update(predictions, labels)

        # Compute the final accuracy and average loss over the entire dataset
        final_accuracy = self.accuracy_metric.compute()
        test_loss = total_loss / len(self.test_dataloader)

        self.model.train()

        # Optionally, return the metrics for further processing
        return final_accuracy, test_loss

    def make_evaluation_step(self, iteration, total_iterations, eval_interval, running_loss, metrics_table, iter_title):
        train_accuracy = self.accuracy_metric.compute()
        average_loss = running_loss / eval_interval

        # Evaluate the model performance on the test dataset
        test_accuracy, test_loss = self.evaluate()  # Assuming evaluate returns accuracy and loss

        # Add row to the metrics table
        metrics_table.add_row(
            [f"{iteration + 1}/{total_iterations}", f"{average_loss:.4f}", f"{train_accuracy:.2%}",
             f"{test_loss:.4f}", f"{test_accuracy:.2%}"])

        # When you're ready to log the table, convert it to a string
        table_string = metrics_table.get_string()

        # Append the table string to the log file
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{table_string}\n\n")

        # Optional: Clear the table after logging if you want a fresh table for each interval
        print(metrics_table)
        metrics_table.clear_rows()

        # Collect metrics
        self.metrics_data.append({
            iter_title: iteration,
            "Train Loss": average_loss,
            "Train Accuracy": train_accuracy.item(),
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy.item()
        })

        # Log metrics to wandb
        if self.use_wandb:
            self.wandb.log({"Train Loss": average_loss,
                            "Train Accuracy": train_accuracy,
                            "Test Loss": test_loss,
                            "Test Accuracy": test_accuracy})

        # Check if the test accuracy is the best we've seen so far
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy

            # Save the model
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "best.pth"))

        # Reset running loss and accuracy metric for the next evaluations
        self.accuracy_metric.reset()


    def save_metrics_and_params(self):
        csv_file_path = os.path.join(self.run_directory, 'results.csv')
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        # Check if metrics_data is not empty
        if not self.metrics_data:
            print("No metrics data to save.")
            return

        # Prepend params_dict to each metrics row
        combined_rows = [{**metrics, **self.params} for metrics in self.metrics_data]

        # Determine fieldnames from the first row (params + metrics keys)
        fieldnames = list(combined_rows[0].keys())

        # Write to CSV file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    def make_optimization_step(self, inputs, targets, first_step=False):
        """

        """

    def get_worker_gradient(self, inputs, labels):
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        loss.backward()
        gradient = torch.cat([param.grad.detach().clone().flatten() for param in self.model.parameters()])
        if self.gradient_clip:
            grad_norm = gradient.norm().item()
            if grad_norm > self.gradient_clip:
                gradient.mul_(self.gradient_clip / grad_norm)
        return gradient, loss.item(), output


