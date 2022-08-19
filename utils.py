import os
import yaml
import numpy as np
import torch


class Logger:
    def __init__(self, args):
        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.verbose = args.verbose
        self.val_accuracies = []
        self.test_accuracies = []
        self.best_steps = []
        self.num_runs = args.num_runs
        self.cur_run = None

        print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run):
        self.cur_run = run
        self.val_accuracies.append(0)
        self.test_accuracies.append(0)
        self.best_steps.append(None)
        print(f'Starting run {run}/{self.num_runs}...')

    def update_metrics(self, metrics, step):
        if metrics['val accuracy'] > self.val_accuracies[-1]:
            self.val_accuracies[-1] = metrics['val accuracy']
            self.test_accuracies[-1] = metrics['test accuracy']
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train accuracy: {metrics["train accuracy"]:.4f}, '
                  f'val accuracy: {metrics["val accuracy"]:.4f}, '
                  f'test accuracy: {metrics["test accuracy"]:.4f}')

    def finish_run(self):
        print(f'Finished run {self.cur_run}. '
              f'Best val accuracy: {self.val_accuracies[-1]:.4f}, '
              f'corresponding test accuracy: {self.test_accuracies[-1]:.4f} '
              f'(step {self.best_steps[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_accuracies)
        val_accuracy_mean = np.mean(self.val_accuracies).item()
        val_accuracy_std = np.std(self.val_accuracies, ddof=1).item() if len(self.val_accuracies) > 1 else np.nan
        test_accuracy_mean = np.mean(self.test_accuracies).item()
        test_accuracy_std = np.std(self.test_accuracies, ddof=1).item() if len(self.test_accuracies) > 1 else np.nan

        metrics = {
            'num runs': num_runs,
            'val accuracy mean': val_accuracy_mean,
            'val accuracy std': val_accuracy_std,
            'test accuracy mean': test_accuracy_mean,
            'test accuracy std': test_accuracy_std,
            'val accuracy values': self.val_accuracies,
            'test accuracy values': self.test_accuracies,
            'best steps': self.best_steps
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

        print(f'Finished {num_runs} runs.')
        print(f'Val accuracy mean: {val_accuracy_mean:.4f}')
        print(f'Val accuracy std: {val_accuracy_std:.4f}')
        print(f'Test accuracy mean: {test_accuracy_mean:.4f}')
        print(f'Test accuracy std: {test_accuracy_std:.4f}')

    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler
