# Model initiation for SPANTHER
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

# from .SPANTHER.models import FIENet
from .components import predict_surv, predict_clf, predict_emb
# from .PANTHER.layers import PANTHERBase
from .SPANTHER.models import SPANTHERBase
from utils.file_utils import save_pkl, load_pkl
from utils.proto_utils import check_prototypes
import ipdb

class SPANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, config, mode):
        super(SPANTHER, self).__init__()

        self.config = config
        emb_dim = config.in_dim

        self.emb_dim = emb_dim
        self.heads = config.heads
        self.outsize = config.out_size
        self.load_proto = config.load_proto
        self.proto_path = config.proto_path
        self.mode = mode

        check_prototypes(config.out_size, self.emb_dim, self.load_proto, config.proto_path)
        
        self.spanther = SPANTHERBase(
            input_size=emb_dim,
            output_size = self.outsize,
            num_layers=config.num_layers,
            num_mixtures=config.num_mixtures,
            num_heads=config.heads,
            out_proj_args=config.sigma,
            pooling=config.pooling,
            concat=config.concat
        )

    def representation(self, x):
        """
        Construct structure-aware unsupervised slide representation
        """
        out, qqs = self.spanther(x)
        return {'repr': out, 'qq': qqs}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    @torch.no_grad()
    def unsup_train(self, data_loader, n_samples=100000, init=None, use_cuda=True):
        '''
        unsupervised training of SPANTHER.
        '''
        self.train(False)

        try:
            n_samples_per_batch = (n_samples + len(data_loader) - 1) // len(data_loader)
        except Exception:
            n_samples_per_batch = 1000
        
        device = torch.device('cuda' if use_cuda else 'cpu')
        dataset = data_loader.dataset

        if self.load_proto:
            if self.proto_path.endswith('pkl'):
                weights = load_pkl(self.proto_path)['prototypes'].squeeze()
            elif self.proto_path.endswith('npy'):
                weights = np.load(self.proto_path)
            weights = torch.from_numpy(weights)

            self.spanther.in_head.unsup_train(None, weight=weights)
        else:
            n_sampled = 0
            samples = torch.Tensor(n_samples, self.emb_dim).to(device)
            for i in tqdm(range(len(dataset)), total=len(dataset), desc='Sampling features'):
                batch = dataset.__getitem__(i)
                data = batch["img"].to(device)
                x = data.x
                samples_batch = self.spanther.in_head.sample(x, n_samples_per_batch)
                size = min(samples_batch.shape[0], n_samples - n_sampled)
                samples[n_sampled: n_sampled + size] = samples_batch[:size]
                n_sampled += size

            print("total number of sampled features: {}".format(n_sampled))
            samples = samples[:n_sampled]

            self.spanther.in_head.unsup_train(samples, weight=None)

            del samples

        for i, layer in enumerate(self.spanther.layers):
            print("Training layer {}".format(i + 1))
            n_sampled = 0
            samples = torch.Tensor(n_samples, layer.input_size)

            # (n_nodes, 1024) -> (n_nodes, 64) -> (n_nodes, 512)
            for idx in tqdm(range(len(dataset)), total=len(dataset), desc=f'Training layer {i + 1}'):
                batch = dataset.__getitem__(idx)
                data = batch["img"].to(device)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                deg_sqrt = getattr(data, 'deg_sqrt', None)
                x = self.spanther.representation(i, x, edge_index, edge_attr, deg_sqrt)
                samples_batch = layer.sample(x, edge_index, n_samples_per_batch)
                size = min(samples_batch.shape[0], n_samples - n_sampled)
                samples[n_sampled: n_sampled + size] = samples_batch[:size].detach().cpu()
                n_sampled += size
                del data, x, edge_index, edge_attr, deg_sqrt

            print("total number of sampled features: {}".format(n_sampled))
            samples = samples[:n_sampled]
            layer.unsup_train(samples, init=init)
            del samples

            if layer.out_proj is not None:
                print("Training out_proj of layer {}".format(i + 1))
                n_sampled = 0
                samples = torch.Tensor(n_samples, layer.out_proj.input_size)

                for idx in tqdm(range(len(dataset)), total=len(dataset), desc=f'Training out_proj of layer {i + 1}'):
                    batch = dataset.__getitem__(i)
                    data = batch["img"].to(device)
                    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                    deg_sqrt = getattr(data, 'deg_sqrt', None)
                    x = self.spanther.representation(i, x, edge_index, edge_attr,
                                                     deg_sqrt, before_out_proj=True)
                    samples_batch = layer.out_proj.sample(x, n_samples_per_batch)
                    size = min(samples_batch.shape[0], n_samples - n_sampled)
                    samples[n_sampled: n_sampled + size] = samples_batch[:size].detach().cpu()
                    n_sampled += size

                    del data, x, edge_index, edge_attr, deg_sqrt

                print("total number of sampled features: {}".format(n_sampled))
                samples = samples[:n_sampled]
                layer.out_proj.unsup_train(samples, weight=init)
                del samples

        print("Training the pooling layer")
        if self.spanther.pooling_type == "ot":
            # only needed for OT pooling layer
            n_sampled = 0
            samples = torch.Tensor(n_samples, self.spanther.pooling.in_dim)
            for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Training pooling layer'):
                batch = dataset.__getitem__(i)
                data = batch["img"].to(device)
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                deg_sqrt = getattr(data, 'deg_sqrt', None)
                x = self.spanther.representation(-1, x, edge_index, edge_attr, deg_sqrt)
                samples_batch = self.spanther.pooling.sample(x, edge_index, n_samples_per_batch)
                size = min(samples_batch.shape[0], n_samples - n_sampled)
                samples[n_sampled: n_sampled + size] = samples_batch[:size].detach().cpu()
                n_sampled += size
                del data, x, edge_index, edge_attr, deg_sqrt
            print("total number of sampled features: {}".format(n_sampled))
            samples = samples[:n_sampled]
            self.spanther.pooling.unsup_train(samples, init=init)

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
    