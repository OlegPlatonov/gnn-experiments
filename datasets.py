import dgl
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu'):
        if name not in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
            raise ValueError(f'Dataset {name} is not supported.')

        print('Preparing data...')

        dataset = DglNodePropPredDataset(name, root='data')
        graph, labels = dataset[0]

        graph = graph.int()
        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        graph = graph.to(device)

        labels = labels.squeeze(axis=1)
        labels = labels.to(device)

        split_idx = dataset.get_idx_split()
        split_idx = {split_name: idx.to(device) for split_name, idx in split_idx.items()}

        self.name = name
        self.device = device

        self.graph = graph
        self.node_features = graph.ndata['feat']
        self.labels = labels

        self.train_idx = split_idx['train']
        self.val_idx = split_idx['valid']
        self.test_idx = split_idx['test']

        self.num_node_features = self.node_features.shape[1]
        self.num_targets = dataset.num_classes

        self.evaluator = Evaluator(name)

    def compute_metrics(self, preds):
        train_accuracy = self.evaluator.eval({'y_true': self.labels[self.train_idx].unsqueeze(-1),
                                              'y_pred': preds[self.train_idx].unsqueeze(-1)})['acc']

        val_accuracy = self.evaluator.eval({'y_true': self.labels[self.val_idx].unsqueeze(-1),
                                            'y_pred': preds[self.val_idx].unsqueeze(-1)})['acc']

        test_accuracy = self.evaluator.eval({'y_true': self.labels[self.test_idx].unsqueeze(-1),
                                             'y_pred': preds[self.test_idx].unsqueeze(-1)})['acc']

        return {'train accuracy': train_accuracy, 'val accuracy': val_accuracy, 'test accuracy': test_accuracy}
