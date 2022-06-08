import argparse
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

from model import Model
from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-proteins', 'ogbn-mag',
                                 'squirrel', 'chameleon', 'actor', 'deezer-europe', 'lastfm-asia', 'facebook', 'github',
                                 'twitch-de', 'twitch-en', 'twitch-es', 'twitch-fr', 'twitch-pt', 'twitch-ru',
                                 'flickr', 'yelp', 'cora', 'citeseer', 'pubmed', 'fraud-yelp-chi', 'fraud-amazon',
                                 'airports-usa', 'airports-europe', 'airports-brazil', 'deezer-hr', 'deezer-hu',
                                 'deezer-ro', 'blogcatalog', 'ppi', 'wikipedia'])

    # model architecture
    parser.add_argument('--model', type=str, default='GT', choices=['ResNet', 'GCN', 'SAGE', 'GAT', 'GT'])
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])

    # regularization
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    # label embeddings
    parser.add_argument('--input_labels_proportion', type=float, default=0)
    parser.add_argument('--label_embedding_dim', type=int, default=128)

    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_degree_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_squared_features', default=False, action='store_true')
    parser.add_argument('--use_centrality_features', default=False, action='store_true')
    parser.add_argument('--use_sbm_features', default=False, action='store_true')
    parser.add_argument('--use_rolx_features', default=False, action='store_true')
    parser.add_argument('--use_graphlet_features', default=False, action='store_true')
    parser.add_argument('--use_spectral_features', default=False, action='store_true')
    parser.add_argument('--use_deepwalk_features', default=False, action='store_true')
    parser.add_argument('--use_struc2vec_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')
    parser.add_argument('--sparse_features_to_dense', default=False, action='store_true',
                        help='Convert sparse node features to dense. This requires more memory, '
                             'but can make training faster.')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_data_splits', type=int, default=10,
                        help='Only used for datasets that do not have standard data splits.')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args


def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    cur_train_idx, cur_label_emb_idx = dataset.get_train_idx_and_label_idx_for_train_step()

    with autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features, x_sparse=dataset.sparse_node_features,
                       label_emb_idx=cur_label_emb_idx)
        loss = dataset.loss_fn(input=logits[cur_train_idx], target=dataset.labels[cur_train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()

    label_emb_idx_for_eval = dataset.get_label_idx_for_evaluation()

    with autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features, x_sparse=dataset.sparse_node_features,
                       label_emb_idx=label_emb_idx_for_eval)

    metrics = dataset.compute_metrics(logits)

    return metrics


def main():
    args = get_args()

    dataset = Dataset(name=args.dataset,
                      add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                      num_data_splits=args.num_data_splits,
                      input_labels_proportion=args.input_labels_proportion,
                      device=args.device,
                      use_sgc_features=args.use_sgc_features,
                      use_identity_features=args.use_identity_features,
                      use_degree_features=args.use_degree_features,
                      use_adjacency_features=args.use_adjacency_features,
                      use_adjacency_squared_features=args.use_adjacency_squared_features,
                      use_centrality_features=args.use_centrality_features,
                      use_sbm_features=args.use_sbm_features,
                      use_rolx_features=args.use_rolx_features,
                      use_graphlet_features=args.use_graphlet_features,
                      use_spectral_features=args.use_spectral_features,
                      use_deepwalk_features=args.use_deepwalk_features,
                      use_struc2vec_features=args.use_struc2vec_features,
                      do_not_use_original_features=args.do_not_use_original_features,
                      sparse_features_to_dense=args.sparse_features_to_dense)

    logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        model = Model(model_name=args.model,
                      num_layers=args.num_layers,
                      input_dim=dataset.num_node_features,
                      sparse_input_dim=dataset.num_sparse_node_features,
                      hidden_dim=args.hidden_dim,
                      output_dim=dataset.num_targets,
                      hidden_dim_multiplier=args.hidden_dim_multiplier,
                      num_heads=args.num_heads,
                      normalization=args.normalization,
                      dropout=args.dropout,
                      use_label_embeddings=(args.input_labels_proportion > 0),
                      label_embedding_bag=dataset.multilabel,
                      num_label_embeddings=dataset.num_label_embeddings,
                      label_embedding_dim=args.label_embedding_dim)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
