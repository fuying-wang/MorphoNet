import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import process_surv, process_clf
from .model_configs import ILRAConfig

"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, _ = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat


class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret = self.mha(global_embedding, X)
        return ret.squeeze(1)


class ILRA(nn.Module):
    def __init__(self, config, mode):
    # def __init__(self, num_layers=2, in_dim=768, n_classes=2, hidden_feat=256, num_heads=8, topk=1, ln=False):
        super().__init__()
        self.config = config
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(config.num_layers):
            block = GAB(dim_in=config.in_dim if idx == 0 else config.hidden_feat,
                        dim_out=config.hidden_feat,
                        num_heads=config.num_heads,
                        num_inds=config.topk,
                        ln=config.ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=config.hidden_feat, num_heads=config.num_heads, num_seeds=config.topk, ln=config.ln)

        # classifier
        self.classifier = nn.Linear(in_features=config.hidden_feat, out_features=config.n_classes)
        self.mode = mode

        initialize_weights(self)
        print(f"ilra2~")

    def forward_no_loss(self, h, attn_mask=None):
        for block in self.gab_blocks:
            h = block(h)

        h = self.pooling(h)
        logits = self.classifier(h)

        out = {'logits': logits}

        return out

    def forward(self, h, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)

        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict

        # for block in self.gab_blocks:
        #     x = block(x)

        # feat = self.pooling(x)
        # logits = self.classifier(feat)

        # logits = logits.squeeze(1)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)

        # return logits, Y_prob, Y_hat


# if __name__ == "__main__":
#     model = ILRA(feat_dim=1024, n_classes=2, hidden_feat=256, num_heads=8, topk=1)
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"num of params: {num_params}")

#     x = torch.randn((1, 1600, 1024))
#     logits, prob, y_hat = model(x)
#     print(f"y shape: {logits.shape}")