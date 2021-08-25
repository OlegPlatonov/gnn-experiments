from torch.nn import functional as F
import dgl
from dgl import ops
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu'):
        if name not in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-proteins']:
            raise ValueError(f'Dataset {name} is not supported.')

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
