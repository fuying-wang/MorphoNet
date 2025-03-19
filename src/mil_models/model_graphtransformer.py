import ipdb
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from typing import Callable, Optional, Sequence, Tuple
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, subgraph
from .components import process_surv, process_clf
from .GraphTransformer.ViT import VisionTransformer
from .GraphTransformer.gcn import GCNBlock


class GraphTransformer(nn.Module):
    def __init__(self, config, mode):
        super().__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100
        self.mode = mode

        self.transformer = VisionTransformer(num_classes=config.n_classes, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(1024, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)       # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                                # 100-> 20

    def forward_no_loss(self, graph_data):

        node_feat = graph_data.x
        edge_index = graph_data.edge_index

        if node_feat.size(0) > 10000:
            # downsample graph to 10k nodes
            indices = torch.randperm(node_feat.size(0))[:10000].type_as(edge_index)
            subset = torch.zeros(node_feat.size(0)).to(indices.device)
            subset[indices] = 1
            subset = subset.bool()
            node_feat = node_feat[subset]
            edge_index, _ = subgraph(subset, edge_index, None, relabel_nodes=True)
    
        X = node_feat
        adj = to_dense_adj(edge_index, max_num_nodes=X.size(0))
        mask = torch.ones(1, X.size(0)).to(X.device)
        X = X.unsqueeze(0)
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)

        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)

        # loss
        pooling_loss = mc1 + o1

        return {
            "logits": out,
            "pooling_loss": pooling_loss
        }

    def forward(self, graph_data, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(graph_data)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)
            results_dict["loss"] = results_dict["loss"] + out['pooling_loss']
            log_dict["loss"] = results_dict["loss"].item()
            log_dict["pooling_loss"] = out['pooling_loss'].item()

        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(graph_data)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
            results_dict["loss"] = results_dict["loss"] + out['pooling_loss']
            log_dict["loss"] = results_dict["loss"].item()
            log_dict["pooling_loss"] = out['pooling_loss'].item()
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict