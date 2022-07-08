import argparse
from tqdm import tqdm

import torch
from torch.nn import functional as F

from model import Model
from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    parser.add_argument('--model', type=str, default='GT', choices=['ResNet', 'GCN', 'SAGE', 'GAT', 'GT'])
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args


def train_step(model, dataset, optimizer, scheduler):
    model.train()
    logits = model(graph=dataset.graph, x=dataset.node_features)
    loss = F.cross_entropy(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    logits = model(graph=dataset.graph, x=dataset.node_features)
    preds = logits.argmax(axis=1)
    metrics = dataset.compute_metrics(preds)

    return metrics


def main():
    args = get_args()
    logger = Logger(args)

    dataset = Dataset(name=args.dataset, add_self_loops=(args.model in ['GCN', 'GAT', 'GT']), device=args.device)

    for run in range(1, args.num_runs + 1):
        model = Model(model_name=args.model,
                      num_layers=args.num_layers,
                      input_dim=dataset.num_node_features,
                      hidden_dim=args.hidden_dim,
                      output_dim=dataset.num_targets,
                      hidden_dim_multiplier=args.hidden_dim_multiplier,
                      num_heads=args.num_heads,
                      normalization=args.normalization,
                      dropout=args.dropout)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler)
                metrics = evaluate(model=model, dataset=dataset)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

        logger.finish_run()
        model.cpu()

    logger.save_metrics()


if __name__ == '__main__':
    main()
