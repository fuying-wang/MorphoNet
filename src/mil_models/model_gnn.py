import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Sequence, Tuple
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from .components import process_surv, process_clf


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        # For a binary mask, a ``True`` value indicates that the
        # corresponding position is not allowed to attend.
        if attn_mask is not None:
            attn_mask = ~attn_mask.bool()
        out = self.attn(self._repeat(q, N), x, x, need_weights=False, key_padding_mask=attn_mask)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
    

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class GNN(nn.Module):
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode

        self.preprocess = nn.Sequential(
            nn.Linear(config.in_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.num_layers = config.num_layers
        if self.num_layers > 0:
            self.convs_list = []
            for _ in range(self.num_layers):
                self.convs_list.append(GATv2Conv(
                    in_channels=config.embed_dim,
                    out_channels=config.embed_dim))
            
            self.convs_list = nn.ModuleList(self.convs_list)
        else:
            self.convs_list = nn.Identity()

        self.pooling = config.pooling
        if self.pooling == "attention":
            self.attention_net = Attn_Net_Gated(
                L=config.embed_dim, D=config.embed_dim, dropout=False, n_classes=1)
        elif self.pooling == "transformer":
            # learnable positional encoding ...
            # self.pe = nn.Parameter(torch.randn(1, 1, config.embed_dim))
            self.attention_net = AttentionalPooler(
                d_model=config.embed_dim, context_dim=config.embed_dim, n_head=8, n_queries=1)
        self.fc = nn.Linear(config.embed_dim, config.n_classes)

    def forward_no_loss(self, graph_data):
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.preprocess(x)
        if self.num_layers > 0:
            for conv in self.convs_list:
                x = conv(x, edge_index)
        else:
            # in this case, it is identity
            x = self.convs_list(x)
        if self.pooling == "attention":
            A, x = self.attention_net(x)
            A = torch.transpose(A, 0, 1)
            A = F.softmax(A, dim=1)
            x = torch.mm(A, x)
        elif self.pooling == "mean":
            x = global_mean_pool(x, graph_data.batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, graph_data.batch)
        elif self.pooling == "transformer":
            # attention pooler
            # note that we don't have pos embedding here
            x = self.attention_net(x.unsqueeze(0))[0]
        else:
            raise NotImplementedError("Not Implemented!")
        logits = self.fc(x)
        out = {"logits": logits}
        return out

    def forward(self, graph_data, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(graph_data)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)

        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(graph_data)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict


if __name__ == "__main__":
    attn_pooler = AttentionalPooler(
        d_model=256, context_dim=128, n_head=8, n_queries=1)
    in_data = torch.randn(10, 256)
    out = attn_pooler(in_data)
    print(out.shape)