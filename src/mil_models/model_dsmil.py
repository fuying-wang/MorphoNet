import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .components import process_surv, process_clf
from .model_configs import DSMILConfig

"""
Li, B., Li, Y. and Eliceiri, K.W., 2021. 
Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. 
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 14318-14328).
"""

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, feats):
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0):  # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, config, mode):
    # def __init__(self, feat_dim=2048, n_classes=2):
        super(DSMIL, self).__init__()
        self.config = config
        self.i_classifier = IClassifier(feature_size=config.in_dim, output_class=config.n_classes)
        self.b_classifier = BClassifier(input_size=config.in_dim, output_class=config.n_classes)
        self.mode = mode

        initialize_weights(self)
        print(f"dsmil~")

    def forward_no_loss(self, h, attn_mask=None):
        h = h.squeeze()
        feats, classes = self.i_classifier(h)
        logits, A, B = self.b_classifier(feats, classes)

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
    
        # feat = feat.squeeze()
        # feats, classes = self.i_classifier(feat)
        # logits, A, B = self.b_classifier(feats, classes)

        # # print(f"logits shape: {logits.shape}")

        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)

        # return logits, classes, Y_prob, Y_hat


# if __name__ == "__main__":
#     bag_size = 103
#     feat_dim = 2048
#     x = torch.randn((bag_size, feat_dim))
#     net = DSMIL()
#     logits, classes, Y_prob, Y_hat = net(x)

#     print(f"class: {classes.shape} predict: {logits}")

#     criterion = nn.CrossEntropyLoss()
#     label = torch.tensor([1]).long()

#     max_prediction, index = torch.max(classes, 0)
#     max_prediction = max_prediction.view(1, -1)
#     print(f"max pred: {max_prediction}")

#     print(logits, label)

#     loss_bag = criterion(logits, label)

#     loss_max = criterion(max_prediction, label)
#     loss = 0.5 * loss_bag + 0.5 * loss_max