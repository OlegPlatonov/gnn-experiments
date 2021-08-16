import torch
from torch.nn import functional as F
import dgl
from dgl import ops
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


class Dataset:
    def __init__(self, name, add_self_loops=False, input_labels_proportion=0, device='cpu'):
        if name not in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-proteins']:
            raise ValueError(f'Dataset {name} is not supported.')

        if name == 'ogbn-proteins' and input_labels_proportion > 0:
            raise ValueError('Label embeddings are not supported for multilabel classification task. '
                             'input_labels_proportion should be set to 0.')

        print('Preparing data...')

        dataset = DglNodePropPredDataset(name, root='data')
        graph, labels = dataset[0]
        graph = graph.int()

        if name == 'ogbn-proteins':
            print("ogbn-proteins graph does not have node features, but it has edge features. "
                  "Node features will be created as mean of edge features of the node's incident edges.")

            graph.ndata['feat'] = ops.copy_e_mean(graph, graph.edata['feat'])

        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        graph = graph.to(device)

        multilabel = (name == 'ogbn-proteins')

        labels = labels.float() if multilabel else labels.squeeze(axis=1)
        labels = labels.to(device)

        split_idx = dataset.get_idx_split()
        split_idx = {split_name: idx.to(device) for split_name, idx in split_idx.items()}

        self.name = name
        self.multilabel = multilabel
        self.device = device

        self.graph = graph
        self.node_features = graph.ndata['feat']
        self.labels = labels

        self.train_idx = split_idx['train']
        self.val_idx = split_idx['valid']
        self.test_idx = split_idx['test']

        self.num_node_features = self.node_features.shape[1]
        self.num_targets = dataset.num_tasks if multilabel else dataset.num_classes

        self.loss_fn = F.binary_cross_entropy_with_logits if multilabel else F.cross_entropy

        self.metric = 'ROC AUC' if multilabel else 'accuracy'
        self.ogb_metric = 'rocauc' if self.metric == 'ROC AUC' else 'acc'

        self.evaluator = Evaluator(name)

        self.input_labels_proportion = input_labels_proportion

    def get_train_idx_and_label_idx_for_train_step(self):
        if self.input_labels_proportion == 0:
            return self.train_idx, None

        n = len(self.train_idx)
        num_input_labels = int(self.input_labels_proportion * n)
        train_mask = (torch.randperm(n, device=self.device) < num_input_labels)

        cur_train_idx = self.train_idx[torch.where(~train_mask)]

        full_mask = torch.zeros_like(self.labels, dtype=torch.bool, device=self.device)
        full_mask[self.train_idx] = train_mask

        cur_label_emb_idx = (self.labels + 1) * full_mask

        return cur_train_idx, cur_label_emb_idx

    def get_label_idx_for_evaluation(self):
        if self.input_labels_proportion == 0:
            return None

        label_emb_idx_for_eval = torch.zeros_like(self.labels, device=self.device)
        label_emb_idx_for_eval[self.train_idx] = self.labels[self.train_idx] + 1

        return label_emb_idx_for_eval

    def compute_metrics(self, logits):
        preds = logits if self.multilabel else logits.argmax(axis=1, keepdims=True)
        labels = self.labels if self.multilabel else self.labels.unsqueeze(1)

        train_metric = self.evaluator.eval({'y_true': labels[self.train_idx],
                                            'y_pred': preds[self.train_idx]})[self.ogb_metric]

        val_metric = self.evaluator.eval({'y_true': labels[self.val_idx],
                                          'y_pred': preds[self.val_idx]})[self.ogb_metric]

        test_metric = self.evaluator.eval({'y_true': labels[self.test_idx],
                                           'y_pred': preds[self.test_idx]})[self.ogb_metric]

        if self.multilabel:
            train_metric = train_metric.item()
            val_metric = val_metric.item()
            test_metric = test_metric.item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics
