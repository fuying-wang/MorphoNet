
"""
Model definition of DeepAttnMISL

If this work is useful for your research, please consider to cite our papers:

[1] "Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks"
Jiawen Yao, XinliangZhu, Jitendra Jonnagaddala, NicholasHawkins, Junzhou Huang,
Medical Image Analysis, Available online 19 July 2020, 101789

[2] "Deep Multi-instance Learning for Survival Prediction from Whole Slide Images", In MICCAI 2019

"""
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
import ipdb

from .components import process_surv, process_clf
# from .model_configs import DeepAttnMILConfig


class DeepAttnMISL(nn.Module):
    """
    Deep AttnMISL Model definition
    """
    def __init__(self, config, mode):
        super(DeepAttnMISL, self).__init__()
        self.config = config
        self.embedding_net = nn.Sequential(nn.Conv2d(1024, 64, 1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1,1))
                                     )


        self.attention = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, config.n_classes)
        )
        self.cluster_num = config.cluster_num
        self.mode = mode


    def masked_softmax(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / (mask+1e-5))
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

    # def cluster(self, x):
    #     ''' clustering '''
    #     assert len(x) == 1, "Only support one WSI"
    #     x = x[0]
        
    #     kmeans = faiss.Kmeans(x.shape[1], 
    #                           self.cluster_num, 
    #                           niter=5, 
    #                           nredo=50,
    #                           verbose=False, 
    #                           max_points_per_centroid=100000,
    #                           gpu=1)
    #     kmeans.train(x.cpu().numpy())
    #     labels = kmeans.assign(x.cpu().numpy())[1]
    #     cluster_feats = []
    #     for i in range(self.cluster_num):
    #         class_i_index = torch.from_numpy(np.where(labels == i)[0]).type_as(x).long()
    #         cluster_feats.append(x[class_i_index].unsqueeze(0))
    #     return cluster_feats

    def forward_no_loss(self, x, attn_mask=None):
        " x is a tensor list"
        res = []
        for i in range(self.cluster_num):
            hh = x[i]
            try:
                hh = rearrange(hh, 'b h w c -> b c h w')
            except:
                ipdb.set_trace()
            output = self.embedding_net(hh)
            output = output.view(output.size()[0], -1)
            res.append(output)

        h = torch.cat(res)

        b = h.size(0)
        c = h.size(1)

        h = h.view(b, c)

        A = self.attention(h)
        A = torch.transpose(A, 1, 0)  # KxN

        A = self.masked_softmax(A, attn_mask)


        M = torch.mm(A, h)  # KxL

        Y_pred = self.fc6(M)

        return {'logits': Y_pred}


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
    
    # def forward(self, x, mask):
        # " x is a tensor list"
        # res = []
        # for i in range(self.cluster_num):
        #     hh = x[i]
        #     output = self.embedding_net(hh)
        #     output = output.view(output.size()[0], -1)
        #     res.append(output)


        # h = torch.cat(res)

        # b = h.size(0)
        # c = h.size(1)

        # h = h.view(b, c)

        # A = self.attention(h)
        # A = torch.transpose(A, 1, 0)  # KxN

        # A = self.masked_softmax(A, mask)


        # M = torch.mm(A, h)  # KxL

        # Y_pred = self.fc6(M)


        # return Y_pred
