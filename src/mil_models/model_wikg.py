import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

from .components import process_surv, process_clf
from .model_configs import WiKGConfig

class WiKG(nn.Module):
    def __init__(self, config, mode):
    # def __init__(self, dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn'):
        super().__init__()
        self.config = config
        self._fc1 = nn.Sequential(nn.Linear(config.dim_in, config.dim_hidden), nn.LeakyReLU())
        
        self.W_head = nn.Linear(config.dim_hidden, config.dim_hidden)
        self.W_tail = nn.Linear(config.dim_hidden, config.dim_hidden)

        self.scale = config.dim_hidden ** -0.5
        self.topk = config.topk
        self.agg_type = config.agg_type

        self.gate_U = nn.Linear(config.dim_hidden, config.dim_hidden // 2)
        self.gate_V = nn.Linear(config.dim_hidden, config.dim_hidden // 2)
        self.gate_W = nn.Linear(config.dim_hidden // 2, config.dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(config.dim_hidden, config.dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(config.dim_hidden * 2, config.dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(config.dim_hidden, config.dim_hidden)
            self.linear2 = nn.Linear(config.dim_hidden, config.dim_hidden)
        else:
            raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(config.dropout)

        self.norm = nn.LayerNorm(config.dim_hidden)
        self.fc = nn.Linear(config.dim_hidden, config.n_classes)

        if config.pool == "mean":
            self.readout = global_mean_pool 
        elif config.pool == "max":
            self.readout = global_max_pool 
        elif config.pool == "attn":
            att_net=nn.Sequential(nn.Linear(config.dim_hidden, config.dim_hidden // 2), nn.LeakyReLU(), nn.Linear(config.dim_hidden//2, 1))     
            self.readout = GlobalAttention(att_net)
        self.mode = mode

        print(f"wikg~")

    def forward_no_loss(self, x, attn_mask=None):
        # try:
        #     x = x["feature"]
        # except:
        #     x = x
        x = self._fc1(x)    # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        
        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)
        logits = self.fc(h)

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

            
# if __name__ == "__main__":
#     data = torch.randn((1, 10000, 384)).cuda()
#     model = WiKG(dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn').cuda()
#     output = model(data)
#     print(output.shape)



