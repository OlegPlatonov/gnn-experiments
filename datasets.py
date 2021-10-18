import os

import torch
from torch.nn import functional as F
import dgl
from dgl import ops

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch_geometric import datasets as pyg_datasets

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from features import (get_sbm_groups,
                      compute_rolx_features,
                      compute_graphlet_degree_vectors,
                      transform_graphlet_degree_vectors_to_binary_features)


class Dataset:
    ogb_dataset_names = ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-proteins']
    pyg_dataset_names = ['squirrel', 'chameleon', 'actor', 'deezer-europe', 'lastfm-asia', 'facebook', 'github',
                         'twitch-de', 'twitch-en', 'twitch-es', 'twitch-fr', 'twitch-pt', 'twitch-ru', 'flickr', 'yelp']

    def __init__(self, name, add_self_loops=False, num_data_splits=None, input_labels_proportion=0,
                 use_sgc_features=False, use_sbm_features=False, use_rolx_features=False, use_graphlet_features=False,
                 device='cpu'):

        print('Preparing data...')
        graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list = self.get_data(name, num_data_splits)

        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        multilabel = (name in ['ogbn-proteins', 'yelp'])

        if multilabel:
            num_targets = labels.shape[1]
        else:
            num_classes = len(labels.unique())
            num_targets = 1 if num_classes == 2 else num_classes

        if num_targets == 1 or multilabel:
            labels = labels.float()

        if use_sgc_features:
            sgc_features = self.compute_sgc_features(graph, node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_sbm_features:
            sbm_features = self.get_sbm_features(name, graph)
            node_features = torch.cat([node_features, sbm_features], axis=1)

        if use_rolx_features:
            rolx_features = self.get_rolx_features(name, graph)
            node_features = torch.cat([node_features, rolx_features], axis=1)

        if use_graphlet_features:
            graphlet_features = self.get_graphlet_features(name, graph)
            node_features = torch.cat([node_features, graphlet_features], axis=1)

        graph = graph.to(device)
        node_features = node_features.to(device)
        labels = labels.to(device)

        train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]

        self.name = name
        self.multilabel = multilabel
        self.device = device

        self.graph = graph
        self.node_features = node_features
        self.labels = labels

        self.train_idx_list = train_idx_list
        self.val_idx_list = val_idx_list
        self.test_idx_list = test_idx_list
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 or multilabel else F.cross_entropy

        self.metric = 'ROC AUC' if num_targets == 1 or multilabel else 'accuracy'
        self.ogb_metric = 'rocauc' if self.metric == 'ROC AUC' else 'acc'

        if name in self.ogb_dataset_names:
            self.evaluator = Evaluator(name)

        self.input_labels_proportion = input_labels_proportion
        self.num_label_embeddings = self.num_targets * 2 + 1 if multilabel else num_classes + 1

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    @classmethod
    def get_data(cls, name, num_data_splits=None):
        if name in cls.ogb_dataset_names:
            return cls.get_ogb_data(name)
        elif name in cls.pyg_dataset_names:
            return cls.get_pyg_data(name, num_data_splits)
        else:
            raise ValueError(f'Dataset {name} is not supported.')

    @classmethod
    def get_ogb_data(cls, name):
        dataset = DglNodePropPredDataset(name, root='data')
        graph, labels = dataset[0]
        graph = graph.int()

        if name == 'ogbn-proteins':
            print("ogbn-proteins graph does not have node features, but it has edge features. "
                  "Node features will be created as mean of edge features of the node's incident edges.")

            graph.ndata['feat'] = ops.copy_e_mean(graph, graph.edata['feat'])

        else:
            labels = labels.squeeze(1)

        node_features = graph.ndata['feat']

        split_idx = dataset.get_idx_split()
        train_idx_list = [split_idx['train']]
        val_idx_list = [split_idx['valid']]
        test_idx_list = [split_idx['test']]

        return graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list

    @classmethod
    def get_pyg_data(cls, name, num_data_splits=None):
        dataset = cls.get_pyg_dataset(name)
        pyg_graph = dataset[0]

        source_nodes, target_nodes = pyg_graph.edge_index
        dgl_graph = dgl.graph((source_nodes, target_nodes), num_nodes=len(pyg_graph.x), idtype=torch.int)
        node_features = pyg_graph.x
        labels = pyg_graph.y

        if name == 'flickr':
            one_hot_encoder = OneHotEncoder(sparse=False, dtype='float32')
            node_features = one_hot_encoder.fit_transform(node_features)
            node_features = torch.tensor(node_features)
        elif name == 'yelp':
            node_features -= node_features.mean(axis=0)
            node_features /= node_features.std(axis=0)

        train_idx_list, val_idx_list, test_idx_list = cls.get_pyg_data_split_idx_lists(name, pyg_graph, num_data_splits)

        return dgl_graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_pyg_dataset(name):
        default_root = os.path.join('data', name)
        if name in ['squirrel', 'chameleon']:
            dataset = pyg_datasets.WikipediaNetwork(root='data', name=name, geom_gcn_preprocess=True)
        elif name == 'actor':
            dataset = pyg_datasets.Actor(root=default_root)
        elif name == 'deezer-europe':
            dataset = pyg_datasets.DeezerEurope(root=default_root)
        elif name == 'lastfm-asia':
            dataset = pyg_datasets.LastFMAsia(root=default_root)
        elif name == 'facebook':
            dataset = pyg_datasets.FacebookPagePage(root=default_root)
        elif name == 'github':
            dataset = pyg_datasets.GitHub(root=default_root)
        elif name in ['twitch-de', 'twitch-en', 'twitch-es', 'twitch-fr', 'twitch-pt', 'twitch-ru']:
            country = name.split('-')[1].upper()
            dataset = pyg_datasets.Twitch(root=os.path.join('data', 'twitch'), name=country)
        elif name == 'flickr':
            dataset = pyg_datasets.Flickr(root=default_root)
        elif name == 'yelp':
            dataset = pyg_datasets.Yelp(root=default_root)
        else:
            raise ValueError(f'Dataset {name} is not supported.')

        return dataset

    @staticmethod
    def get_pyg_data_split_idx_lists(name, pyg_graph, num_data_splits=None):
        if name in ['flickr', 'yelp']:
            train_idx_list = [torch.where(pyg_graph.train_mask)[0]]
            val_idx_list = [torch.where(pyg_graph.val_mask)[0]]
            test_idx_list = [torch.where(pyg_graph.test_mask)[0]]

        elif name in ['squirrel', 'chameleon', 'actor']:
            num_splits = pyg_graph.train_mask.shape[1]
            train_idx_list = [torch.where(pyg_graph.train_mask[:, i])[0] for i in range(num_splits)]
            val_idx_list = [torch.where(pyg_graph.val_mask[:, i])[0] for i in range(num_splits)]
            test_idx_list = [torch.where(pyg_graph.test_mask[:, i])[0] for i in range(num_splits)]

        else:
            if num_data_splits is None:
                raise ValueError(f'Dataset {name} does not have standard data splits. '
                                 'num_data_splits should be provided.')

            train_idx_list, val_idx_list, test_idx_list = [], [], []

            full_idx = torch.arange(len(pyg_graph.y))

            for i in range(num_data_splits):
                train_idx, val_and_test_idx = train_test_split(full_idx, test_size=0.5, random_state=i,
                                                               stratify=pyg_graph.y)

                val_idx, test_idx = train_test_split(val_and_test_idx, test_size=0.5, random_state=i,
                                                     stratify=pyg_graph.y[val_and_test_idx])

                train_idx_list.append(train_idx.sort()[0])
                val_idx_list.append(val_idx.sort()[0])
                test_idx_list.append(test_idx.sort()[0])

        return train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features

    @classmethod
    def get_sbm_features(cls, name, graph):
        data_dir = cls.get_data_dir(name)
        file = os.path.join(data_dir, 'sbm_groups.pt')
        if os.path.isfile(file):
            sbm_groups = torch.load(file)
        else:
            print('Fitting the SBM...')
            sbm_groups = get_sbm_groups(graph)
            torch.save(sbm_groups, file)
            print(f'SBM groups were saved to {file}.')

        sbm_features = F.one_hot(sbm_groups)

        return sbm_features

    @classmethod
    def get_rolx_features(cls, name, graph):
        data_dir = cls.get_data_dir(name)
        file = os.path.join(data_dir, 'rolx_features.pt')
        if os.path.isfile(file):
            rolx_features = torch.load(file)
        else:
            print('Computing RolX features...')
            rolx_features = compute_rolx_features(graph)
            torch.save(rolx_features, file)
            print(f'RolX features were saved to {file}.')

        return rolx_features

    @classmethod
    def get_graphlet_features(cls, name, graph):
        data_dir = cls.get_data_dir(name)
        file = os.path.join(data_dir, 'graphlet_degree_vectors.pt')
        if os.path.isfile(file):
            graphlet_degree_vectors = torch.load(file)
        else:
            print('Computing graphlet degree vectors...')
            graphlet_degree_vectors = compute_graphlet_degree_vectors(graph)
            torch.save(graphlet_degree_vectors, file)
            print(f'Graphlet degree vectors were saved to {file}.')

        graphlet_features = transform_graphlet_degree_vectors_to_binary_features(graphlet_degree_vectors)

        return graphlet_features

    @staticmethod
    def get_data_dir(name):
        if name in Dataset.ogb_dataset_names:
            name = name.replace('-', '_')
            return os.path.join('data', name)
        elif name in ['squirrel', 'chameleon']:
            return os.path.join('data', name, 'geom_gcn')
        elif name in ['twitch-de', 'twitch-en', 'twitch-es', 'twitch-fr', 'twitch-pt', 'twitch-ru']:
            country = name.split('-')[1].upper()
            return os.path.join('data', 'twitch', country)
        else:
            return os.path.join('data', name)

    def get_label_embeddings_idx(self, labels):
        if self.multilabel:
            return torch.arange(start=1, end=self.num_label_embeddings, step=2, device=self.device) + labels.long()
        else:
            return labels.long() + 1

    def get_train_idx_and_label_idx_for_train_step(self):
        if self.input_labels_proportion == 0:
            return self.train_idx, None

        n = len(self.train_idx)
        num_input_labels = int(self.input_labels_proportion * n)
        train_mask = (torch.randperm(n, device=self.device) < num_input_labels)

        cur_train_idx = self.train_idx[torch.where(~train_mask)]

        if self.multilabel:
            train_mask = train_mask.unsqueeze(1)

        full_mask = torch.zeros_like(self.labels, dtype=torch.bool, device=self.device)
        full_mask[self.train_idx] = train_mask

        cur_label_emb_idx = self.get_label_embeddings_idx(self.labels) * full_mask

        return cur_train_idx, cur_label_emb_idx

    def get_label_idx_for_evaluation(self):
        if self.input_labels_proportion == 0:
            return None

        label_emb_idx_for_eval = torch.zeros_like(self.labels, dtype=torch.long, device=self.device)
        label_emb_idx_for_eval[self.train_idx] = self.get_label_embeddings_idx(self.labels[self.train_idx])

        return label_emb_idx_for_eval

    def compute_metrics(self, logits):
        if self.name in self.ogb_dataset_names:
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

        else:
            if self.num_targets == 1 or self.multilabel:
                train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                             y_score=logits[self.train_idx].cpu().numpy(),
                                             average='macro').item()

                val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                           y_score=logits[self.val_idx].cpu().numpy(),
                                           average='macro').item()

                test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                            y_score=logits[self.test_idx].cpu().numpy(),
                                            average='macro').item()

            else:
                preds = logits.argmax(axis=1)
                train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
                val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
                test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics
