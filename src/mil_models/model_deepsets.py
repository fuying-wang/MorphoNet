import torch
import torch.nn as nn

from .components import predict_surv, predict_clf, predict_emb

class DeepSets(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out = torch.mean(x, dim=1) # B x N x L --> B x L

        return {'repr': out}
    
    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y