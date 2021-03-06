import os
import urllib

import torch
from torch.nn import functional as F
import dgl
from dgl import ops

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch_geometric import datasets as pyg_datasets

from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from features import (compute_centrality_measures, get_sbm_groups, compute_rolx_features,
                      compute_graphlet_degree_vectors, transform_graphlet_degree_vectors_to_binary_features)


class Dataset:
    ogb_dataset_names = ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-proteins', 'ogbn-mag']
    pyg_dataset_names = ['squirrel', 'chameleon', 'actor', 'deezer-europe', 'lastfm-asia', 'facebook', 'github',
                         'twitch-de', 'twitch-en', 'twitch-es', 'twitch-fr', 'twitch-pt', 'twitch-ru', 'flickr', 'yelp',
                         'cora', 'citeseer', 'pubmed', 'coauthor-cs', 'coauthor-physics', 'amazon-computers',
                         'amazon-photo', 'airports-usa', 'airports-europe', 'airports-brazil', 'deezer-hr', 'deezer-hu',
                         'deezer-ro']
    dgl_dataset_names = ['fraud-yelp-chi', 'fraud-amazon']
    other_dataset_names = ['blogcatalog', 'ppi', 'wikipedia']

    multilabel_names = ['ogbn-proteins', 'yelp', 'deezer-hr', 'deezer-hu', 'deezer-ro', 'blogcatalog', 'ppi',
                        'wikipedia']

    no_features_names = ['airports-usa', 'airports-europe', 'airports-brazil', 'deezer-hr', 'deezer-hu', 'deezer-ro',
                         'blogcatalog', 'ppi', 'wikipedia']

    def __init__(self, name, add_self_loops=False, num_data_splits=None, input_labels_proportion=0, device='cpu',
                 use_sgc_features=False, use_identity_features=False, use_degree_features=False,
                 use_adjacency_features=False, use_adjacency_squared_features=False, use_centrality_features=False,
                 use_sbm_features=False, use_rolx_features=False, use_graphlet_features=False,
                 use_spectral_features=False, use_deepwalk_features=False, use_struc2vec_features=False,
                 do_not_use_original_features=False, sparse_features_to_dense=False):

        additional_features = [use_sgc_features, use_identity_features, use_degree_features, use_adjacency_features,
                               use_adjacency_squared_features, use_centrality_features, use_sbm_features,
                               use_rolx_features, use_graphlet_features, use_spectral_features,
                               use_deepwalk_features, use_struc2vec_features]

        if name in self.no_features_names:
            if use_sgc_features:
                raise ValueError('SGC features cannot be used for datasets without node features. '
                                 'The argument use_sgc_features should be omitted.')

            if not any(additional_features[1:]):
                raise ValueError('For datasets without node features at least one of the arguments '
                                 'use_identity_features, use_degree_features, use_adjacency_features, '
                                 'use_adjacency_squared_features, use_centrality_features, use_sbm_features, '
                                 'use_rolx_features, use_graphlet_features, use_spectral_features, '
                                 'use_deepwalk_features, use_struc2vec_features should be used.')

        if do_not_use_original_features and not any(additional_features):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_degree_features, use_adjacency_features, '
                             'use_adjacency_squared_features, use_centrality_features, use_sbm_features, '
                             'use_rolx_features, use_graphlet_features, use_spectral_features, '
                             'use_deepwalk_features, use_struc2vec_features should be used.')

        print('Preparing data...')
        graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list = self.get_data(name, num_data_splits)

        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)
        graph = dgl.remove_self_loop(graph)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        multilabel = (name in self.multilabel_names)

        if multilabel:
            num_targets = labels.shape[1]
        else:
            num_classes = len(labels.unique())
            num_targets = 1 if num_classes == 2 else num_classes

        if num_targets == 1 or multilabel:
            labels = labels.float()

        node_features, sparse_node_features = self.augment_node_features(
            name=name,
            graph=graph,
            node_features=node_features,
            use_sgc_features=use_sgc_features,
            use_identity_features=use_identity_features,
            use_degree_features=use_degree_features,
            use_adjacency_features=use_adjacency_features,
            use_adjacency_squared_features=use_adjacency_squared_features,
            use_centrality_features=use_centrality_features,
            use_sbm_features=use_sbm_features,
            use_rolx_features=use_rolx_features,
            use_graphlet_features=use_graphlet_features,
            use_spectral_features=use_spectral_features,
            use_deepwalk_features=use_deepwalk_features,
            use_struc2vec_features=use_struc2vec_features,
            do_not_use_original_features=do_not_use_original_features,
            sparse_features_to_dense=sparse_features_to_dense
        )

        graph = graph.to(device)
        node_features = node_features.to(device)
        sparse_node_features = sparse_node_features.to(device)
        labels = labels.to(device)

        train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]

        self.name = name
        self.multilabel = multilabel
        self.device = device

        self.graph = graph
        self.node_features = node_features if node_features.shape[1] > 0 else None
        self.sparse_node_features = sparse_node_features if sparse_node_features.shape[1] > 0 else None
        self.labels = labels

        self.train_idx_list = train_idx_list
        self.val_idx_list = val_idx_list
        self.test_idx_list = test_idx_list
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_sparse_node_features = sparse_node_features.shape[1]
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

    @staticmethod
    def get_data(name, num_data_splits=None):
        if name in Dataset.ogb_dataset_names:
            return Dataset.get_ogb_data(name)
        elif name in Dataset.pyg_dataset_names:
            return Dataset.get_pyg_data(name, num_data_splits)
        elif name in Dataset.dgl_dataset_names:
            return Dataset.get_dgl_data(name, num_data_splits)
        elif name in Dataset.other_dataset_names:
            return Dataset.get_other_data(name, num_data_splits)
        else:
            raise ValueError(f'Dataset {name} is not supported.')

    @staticmethod
    def get_ogb_data(name):
        dataset = DglNodePropPredDataset(name, root='data')
        graph, labels = dataset[0]
        graph = graph.int()
        split_idx = dataset.get_idx_split()

        if name == 'ogbn-mag':
            print('ogbn-mag is a heterogeneous graph, but only the subgraph with paper nodes and citation relations '
                  'will be used.')

            node_features = graph.ndata['feat']['paper']
            graph = dgl.graph(graph.edges(etype='cites'), num_nodes=graph.num_nodes('paper'), idtype=torch.int)
            graph.ndata['feat'] = node_features
            labels = labels['paper']
            split_idx = {key: value['paper'] for key, value in split_idx.items()}

        if name == 'ogbn-proteins':
            print("ogbn-proteins graph does not have node features, but it has edge features. "
                  "Node features will be created as mean of edge features of the node's incident edges.")

            graph.ndata['feat'] = ops.copy_e_mean(graph, graph.edata['feat'])

        else:
            labels = labels.squeeze(1)

        node_features = graph.ndata['feat']

        train_idx_list = [split_idx['train']]
        val_idx_list = [split_idx['valid']]
        test_idx_list = [split_idx['test']]

        return graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_pyg_data(name, num_data_splits=None):
        dataset = Dataset.get_pyg_dataset(name)
        pyg_graph = dataset[0]

        source_nodes, target_nodes = pyg_graph.edge_index
        n = len(pyg_graph.y)
        dgl_graph = dgl.graph((source_nodes, target_nodes), num_nodes=n, idtype=torch.int)
        node_features = pyg_graph.x if name not in Dataset.no_features_names else torch.tensor([[] for _ in range(n)])
        labels = pyg_graph.y

        if name == 'flickr':
            node_features = Dataset.one_hot_encode_features(node_features)
        elif name == 'yelp':
            node_features = Dataset.normalize_features(node_features)

        if name in ['deezer-hr', 'deezer-hu', 'deezer-ro']:
            labels = Dataset.drop_rare_labels(labels, min_label_count=1000)

        train_idx_list, val_idx_list, test_idx_list = Dataset.get_pyg_data_split_idx_lists(
            name=name, pyg_graph=pyg_graph, num_data_splits=num_data_splits
        )

        return dgl_graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_dgl_data(name, num_data_splits):
        if name == 'fraud-yelp-chi':
            dataset = dgl.data.FraudYelpDataset(raw_dir='data/fraud-yelp-chi')
        elif name == 'fraud-amazon':
            dataset = dgl.data.FraudAmazonDataset(raw_dir='data/fraud-amazon')
        else:
            raise ValueError(f'Dataset {name} is not supported.')

        print(f'{name} is a heterogeneous graph with several different edge types, but they all will be treated '
              'in the same way.')

        graph = dataset[0]
        graph = graph.int()

        if name == 'fraud-amazon':
            labeled_mask = (graph.ndata['train_mask'] | graph.ndata['val_mask'] | graph.ndata['test_mask'])
            labeled_idx = torch.where(labeled_mask)[0]
        else:
            labeled_idx = None

        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'], store_type=False)

        node_features = graph.ndata['feature'].float()
        labels = graph.ndata['label'].reshape(-1)

        if name == 'fraud-amazon':
            num_unique_feature_values = [len(node_features[:, i].unique()) for i in range(node_features.shape[1])]
            one_hot_idx = [i for i, num in enumerate(num_unique_feature_values) if num <= 5]
            one_hot_encoded_features = Dataset.one_hot_encode_features(node_features[:, one_hot_idx])

            other_idx = [i for i in range(node_features.shape[1]) if i not in one_hot_idx]
            other_features = Dataset.normalize_features(node_features[:, other_idx])

            node_features = torch.cat([one_hot_encoded_features, other_features], axis=1)

        train_idx_list, val_idx_list, test_idx_list = Dataset.get_random_data_split_idx_lists(
            name=name, num_data_splits=num_data_splits, labels=labels, labeled_idx=labeled_idx
        )

        return graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_other_data(name, num_data_splits):
        if name == 'blogcatalog':
            filename = os.path.join('data', name, 'blogcatalog.mat')
            url = 'http://leitang.net/code/social-dimension/data/blogcatalog.mat'
        elif name == 'ppi':
            filename = os.path.join('data', name, 'Homo_sapiens.mat')
            url = 'http://snap.stanford.edu/node2vec/Homo_sapiens.mat'
        elif name == 'wikipedia':
            filename = os.path.join('data', name, 'POS.mat')
            url = 'http://snap.stanford.edu/node2vec/POS.mat'
        else:
            raise ValueError(f'Dataset {name} is not supported.')

        if not os.path.isfile(filename):
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            urllib.request.urlretrieve(url, filename)

        data = loadmat(filename)
        source_nodes, target_nodes = data['network'].nonzero()
        labels = torch.tensor(data['group'].toarray())
        n = len(labels)
        graph = dgl.graph((source_nodes, target_nodes), num_nodes=n, idtype=torch.int)
        node_features = torch.tensor([[] for _ in range(n)])

        labels = Dataset.drop_rare_labels(labels, min_label_count=100)

        train_idx_list, val_idx_list, test_idx_list = Dataset.get_random_data_split_idx_lists(
            name=name, num_data_splits=num_data_splits, labels=labels
        )

        return graph, node_features, labels, train_idx_list, val_idx_list, test_idx_list

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
        elif name in ['cora', 'citeseer', 'pubmed']:
            dataset = pyg_datasets.Planetoid(root=default_root, name=name)
        elif name in ['coauthor-cs', 'coauthor-physics']:
            field = name.split('-')[1]
            dataset = pyg_datasets.Coauthor(root=os.path.join('data', 'coauthor'), name=field)
        elif name in ['amazon-computers', 'amazon-photo']:
            product = name.split('-')[1]
            dataset = pyg_datasets.Amazon(root=os.path.join('data', 'amazon'), name=product)
        elif name in ['airports-usa', 'airports-europe', 'airports-brazil']:
            location = name.split('-')[1]
            dataset = pyg_datasets.Airports(root=os.path.join('data', 'airports'), name=location)
        elif name in ['deezer-hr', 'deezer-hu', 'deezer-ro']:
            country = name.split('-')[1].upper()
            dataset = pyg_datasets.GemsecDeezer(root=os.path.join('data', 'gemsec-deezer'), name=country)
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
            train_idx_list, val_idx_list, test_idx_list = Dataset.get_random_data_split_idx_lists(
                name=name, num_data_splits=num_data_splits, labels=pyg_graph.y
            )

        return train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_random_data_split_idx_lists(name, num_data_splits, labels, labeled_idx=None):
        if num_data_splits is None:
            raise ValueError(f'Dataset {name} does not have standard data splits. '
                             'num_data_splits should be provided.')

        train_idx_list, val_idx_list, test_idx_list = [], [], []

        if labeled_idx is None:
            labeled_idx = torch.arange(len(labels))

        for i in range(num_data_splits):
            stratify = labels[labeled_idx] if name not in Dataset.multilabel_names else None
            train_idx, val_and_test_idx = train_test_split(labeled_idx, test_size=0.5, random_state=i,
                                                           stratify=stratify)

            stratify = labels[val_and_test_idx] if name not in Dataset.multilabel_names else None
            val_idx, test_idx = train_test_split(val_and_test_idx, test_size=0.5, random_state=i,
                                                 stratify=stratify)

            train_idx_list.append(train_idx.sort()[0])
            val_idx_list.append(val_idx.sort()[0])
            test_idx_list.append(test_idx.sort()[0])

        return train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def normalize_features(x):
        x -= x.mean(axis=0)
        x /= x.std(axis=0)

        return x

    @staticmethod
    def one_hot_encode_features(x):
        one_hot_encoder = OneHotEncoder(drop='if_binary', sparse=False, dtype='float32')
        x = one_hot_encoder.fit_transform(x)
        x = torch.tensor(x)

        return x

    @staticmethod
    def drop_rare_labels(labels, min_label_count):
        label_counts = labels.sum(axis=0)
        labels = labels[:, (label_counts >= min_label_count)]

        return labels

    @staticmethod
    def augment_node_features(name, graph, node_features, use_sgc_features, use_identity_features,
                              use_degree_features, use_adjacency_features, use_adjacency_squared_features,
                              use_centrality_features, use_sbm_features, use_rolx_features, use_graphlet_features,
                              use_spectral_features, use_deepwalk_features, use_struc2vec_features,
                              do_not_use_original_features, sparse_features_to_dense):

        n = graph.num_nodes()
        sparse_node_features = torch.sparse_coo_tensor(size=(n, 0))

        original_node_features = node_features
        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = Dataset.compute_sgc_features(graph, original_node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_identity_features:
            indices = torch.vstack([torch.arange(n), torch.arange(n)])
            values = torch.ones(n)
            identity_matrix = torch.sparse_coo_tensor(indices=indices, values=values, size=(n, n))
            sparse_node_features = torch.cat([sparse_node_features, identity_matrix], axis=1)

        if use_degree_features:
            degree_features = Dataset.get_degree_features(graph)
            node_features = torch.cat([node_features, degree_features], axis=1)

        if use_adjacency_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix()
            sparse_node_features = torch.cat([sparse_node_features, adj_matrix], axis=1)

        if use_adjacency_squared_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix()
            adj_matrix_squared = torch.sparse.mm(adj_matrix, adj_matrix)
            sparse_node_features = torch.cat([sparse_node_features, adj_matrix_squared], axis=1)

        if use_centrality_features:
            centrality_features = Dataset.get_centrality_features(name, graph)
            node_features = torch.cat([node_features, centrality_features], axis=1)

        if use_sbm_features:
            sbm_features = Dataset.get_sbm_features(name, graph)
            node_features = torch.cat([node_features, sbm_features], axis=1)

        if use_rolx_features:
            rolx_features = Dataset.get_rolx_features(name, graph)
            node_features = torch.cat([node_features, rolx_features], axis=1)

        if use_graphlet_features:
            graphlet_features = Dataset.get_graphlet_features(name, graph)
            node_features = torch.cat([node_features, graphlet_features], axis=1)

        if use_spectral_features:
            spectral_features = Dataset.get_spectral_features(name)
            node_features = torch.cat([node_features, spectral_features], axis=1)

        if use_deepwalk_features:
            deepwalk_features = Dataset.get_deepwalk_features(name)
            node_features = torch.cat([node_features, deepwalk_features], axis=1)

        if use_struc2vec_features:
            struc2vec_features = Dataset.get_struc2vec_features(name)
            node_features = torch.cat([node_features, struc2vec_features], axis=1)

        if sparse_features_to_dense:
            node_features = torch.cat([node_features, sparse_node_features.to_dense()], axis=1)
            sparse_node_features = torch.sparse_coo_tensor(size=(n, 0))

        return node_features, sparse_node_features

    @staticmethod
    def get_data_dir(name):
        if name in Dataset.ogb_dataset_names:
            name = name.replace('-', '_')
            return os.path.join('data', name)
        elif name == 'fraud-yelp-chi':
            return os.path.join('data', name, 'yelp')
        elif name == 'fraud-amazon':
            return os.path.join('data', name, 'amazon')
        elif name in ['squirrel', 'chameleon']:
            return os.path.join('data', name, 'geom_gcn')
        elif name in ['twitch-de', 'twitch-en', 'twitch-es', 'twitch-fr', 'twitch-pt', 'twitch-ru']:
            country = name.split('-')[1].upper()
            return os.path.join('data', 'twitch', country)
        elif name in ['coauthor-cs', 'coauthor-physics']:
            field = 'CS' if name == 'coauthor-cs' else 'Physics'
            return os.path.join('data', 'coauthor', field)
        elif name in ['amazon-computers', 'amazon-photo']:
            product = 'Computers' if name == 'amazon-computers' else 'Photo'
            return os.path.join('data', 'amazon', product)
        elif name in ['airports-usa', 'airports-europe', 'airports-brazil']:
            location = name.split('-')[1]
            return os.path.join('data', 'airports', location)
        elif name in ['deezer-hr', 'deezer-hu', 'deezer-ro']:
            country = name.split('-')[1].upper()
            return os.path.join('data', 'gemsec-deezer', country)
        else:
            return os.path.join('data', name)

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

    @staticmethod
    def get_degree_features(graph, max_degree=50):
        degrees = graph.out_degrees().long()
        degrees = torch.minimum(degrees, torch.tensor(max_degree))
        degrees_one_hot = F.one_hot(degrees)

        return degrees_one_hot

    @staticmethod
    def get_centrality_features(name, graph):
        data_dir = Dataset.get_data_dir(name)
        file = os.path.join(data_dir, 'centrality_measures.pt')
        if os.path.isfile(file):
            centrality_measures = torch.load(file)
        else:
            print('Computing centrality measures...')
            centrality_measures = compute_centrality_measures(graph)
            torch.save(centrality_measures, file)
            print(f'Centrality measures were saved to {file}.')

        centrality_measures -= centrality_measures.min(axis=0)[0]
        centrality_measures /= centrality_measures.max(axis=0)[0]

        return centrality_measures

    @staticmethod
    def get_sbm_features(name, graph):
        data_dir = Dataset.get_data_dir(name)
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

    @staticmethod
    def get_rolx_features(name, graph):
        data_dir = Dataset.get_data_dir(name)
        file = os.path.join(data_dir, 'rolx_features.pt')
        if os.path.isfile(file):
            rolx_features = torch.load(file)
        else:
            print('Computing RolX features...')
            rolx_features = compute_rolx_features(graph)
            torch.save(rolx_features, file)
            print(f'RolX features were saved to {file}.')

        return rolx_features

    @staticmethod
    def get_graphlet_features(name, graph):
        data_dir = Dataset.get_data_dir(name)
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
    def get_spectral_features(name):
        data_dir = Dataset.get_data_dir(name)
        file = os.path.join(data_dir, 'spectral_embeddings.pt')
        if os.path.isfile(file):
            spectral_embeddings = torch.load(file)
            return spectral_embeddings
        else:
            raise FileNotFoundError(f'File {file} not found. Precompute spectral embeddings or ommit argument '
                                    'use_spectral_features. You can use this repository to precompute spectral '
                                    'embeddings: https://github.com/CUAI/CorrectAndSmooth.')

    @staticmethod
    def get_deepwalk_features(name):
        data_dir = Dataset.get_data_dir(name)
        file = os.path.join(data_dir, 'deepwalk_embeddings.pt')
        if os.path.isfile(file):
            deepwalk_embeddings = torch.load(file)
            return deepwalk_embeddings
        else:
            raise FileNotFoundError(f'File {file} not found. Precompute DeepWalk embeddings or ommit argument '
                                    'use_depwalk_features. You can use this repository to precompute DeepWalk '
                                    'embeddings: https://github.com/phanein/deepwalk.')

    @staticmethod
    def get_struc2vec_features(name):
        data_dir = Dataset.get_data_dir(name)
        file = os.path.join(data_dir, 'struc2vec_embeddings.pt')
        if os.path.isfile(file):
            struc2vec_embeddings = torch.load(file)
            return struc2vec_embeddings
        else:
            raise FileNotFoundError(f'File {file} not found. Precompute struc2vec embeddings or ommit argument '
                                    'use_struc2vec_features. You can use this repository to precompute struc2vec '
                                    'embeddings: https://github.com/leoribeiro/struc2vec.')
