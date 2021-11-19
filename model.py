import torch
from torch import nn
from torch.cuda.amp import autocast
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule,
                     TransformerAttentionModule)


MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GT': [TransformerAttentionModule, FeedForwardModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, sparse_input_dim, hidden_dim, output_dim,
                 hidden_dim_multiplier, num_heads, normalization, dropout, use_label_embeddings=False,
                 label_embedding_bag=False, num_label_embeddings=None, label_embedding_dim=128):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.use_label_embeddings = use_label_embeddings
        if use_label_embeddings:
            input_dim += label_embedding_dim
            if label_embedding_bag:
                self.label_embeddings = nn.EmbeddingBag(num_embeddings=num_label_embeddings,
                                                        embedding_dim=label_embedding_dim,
                                                        mode='mean')
            else:
                self.label_embeddings = nn.Embedding(num_embeddings=num_label_embeddings,
                                                     embedding_dim=label_embedding_dim)

        if input_dim > 0:
            self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        if sparse_input_dim > 0:
            self.sparse_input_linear = nn.Linear(in_features=sparse_input_dim, out_features=hidden_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x, x_sparse=None, label_emb_idx=None):
        if self.use_label_embeddings:
            label_embeddings = self.label_embeddings(label_emb_idx)
            x = torch.cat([x, label_embeddings], axis=1) if x is not None else label_embeddings

        if x_sparse is None:
            x = self.input_linear(x)
        elif x is None:
            with autocast(enabled=False):
                x = self.sparse_input_linear(x_sparse)
        else:
            x = self.input_linear(x)
            with autocast(enabled=False):
                x += self.sparse_input_linear(x_sparse)

        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x
