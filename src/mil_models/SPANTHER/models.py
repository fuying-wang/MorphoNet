# -*- coding: utf-8 -*-
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import ipdb
import numpy as np
from .layers import FIELayer, KernelLayer, FIEPooling
from ..OT.otk.layers import OTKernel


class SPANTHERBase(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, num_mixtures=1, num_heads=1,
                 residue=True, out_proj="kernel", out_proj_args=1.0, use_deg=False,
                 pooling='fie', concat=False, out_type='allcat'):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.concat = concat
        self.out_type = out_type

        # hidden_size = output_size
        self.in_head = KernelLayer(input_size, hidden_size, sigma=out_proj_args)

        layers = []

        for i in range(num_layers - 1):
            layers.append(
                FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
                         residue, out_proj, 'exp', out_proj_args, use_deg)
            )

        layers.append(
            # out_proj becomes None for the last layer
            FIELayer(hidden_size, hidden_size, num_mixtures, num_heads,
                     residue, None, 'exp', out_proj_args, use_deg)
        )

        self.layers = nn.ModuleList(layers)

        self.pooling_type = pooling
        if self.pooling_type == 'mean':
            self.pooling = gnn.global_mean_pool
        elif self.pooling_type == 'sum':
            self.pooling = gnn.global_add_pool
        elif self.pooling_type == 'ot':
            pool_size = hidden_size * (num_mixtures * num_layers + 1) if concat else hidden_size * num_mixtures
            # pool_size = hidden_size
            distance = 'euclidean'
            max_iter = 100
            ot_eps = 0.1
            self.pooling = OTKernel(
                in_dim=pool_size,
                out_size=output_size,
                distance=distance,
                heads=num_heads,
                max_iter=max_iter,
                eps=ot_eps,
                # norm=True
                # TODO: it seems that norm=False achieves better performance 
                norm=False
            )

            if self.out_type == 'allcat':
                self.out_features = input_size * output_size * num_heads
            elif self.out_type == 'weight_avg_mean':
                self.out_features = input_size * num_heads
            else:
                raise NotImplementedError(f"OT Not implemented for {self.out_type}!")

            # if load_proto:
            #     if proto_path.endswith('pkl'):
            #         weights = load_pkl(proto_path)['prototypes'].squeeze()
            #     elif proto_path.endswith('npy'):
            #         weights = np.load(proto_path)
            #     weights = torch.from_numpy(weights)
            #     self.pooling.weight.data.copy_(weights)
        elif self.pooling_type == 'fie':
            pool_size = hidden_size * (num_mixtures * num_layers + 1) if concat else hidden_size * num_mixtures
            # pool_size = hidden_size * num_layers + input_size if concat else hidden_size
            self.pooling = FIEPooling(
                pool_size, hidden_size, num_heads, residue, out_proj, out_proj_args
            )
        else:
            self.pooling = None
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        deg_sqrt = getattr(data, 'deg_sqrt', None)

        # (num_nodes, 1024)
        # try only add in_head layer
        x = self.in_head(x)
        # (num_nodes, 64)

        outputs = [x]
        output = x
        for i, mod in enumerate(self.layers):
            output = mod(output, edge_index, edge_attr, deg_sqrt=deg_sqrt)
            outputs.append(output)
            
        if self.concat:
            output = torch.cat(outputs, dim=-1)
        else:
            output = outputs[-1]

        if self.pooling is not None:
            if self.pooling_type == "ot":
                out = self.pooling(output.unsqueeze(0))
                if self.out_type == 'allcat':
                    out = out.reshape(1, -1)
                elif self.out_type == 'weight_avg_mean':
                    out = torch.mean(out, dim=1)
                else:
                    raise NotImplementedError(f"OTK Not implemented for {self.out_type}!")
                bag_size = out.size(0)
                weights = torch.ones(bag_size, 1).to(out.device) / bag_size
                return out, weights
            elif self.pooling_type == "fie":
                ipdb.set_trace()
                output = self.pooling(output, data.batch)
            else:
                output = self.pooling(output, data.batch)
                bag_size = output.size(0)
                weights = torch.ones(bag_size, 1).to(output.device) / bag_size
                return output, weights

    def representation(self, n, x, edge_index, edge_attr=None, deg_sqrt=None,
                       before_out_proj=False):
        output = []
        x = self.in_head(x)
        if self.concat:
            output.append(x)
        if n == -1:
            n = self.num_layers
        for i in range(n):
            x = self.layers[i](x, edge_index, edge_attr, deg_sqrt=deg_sqrt) # 15, 512
            if self.concat:
                output.append(x)
        if before_out_proj:
            x = self.layers[n](x, edge_index, edge_attr, deg_sqrt=deg_sqrt,
                            before_out_proj=before_out_proj)
            if self.concat:
                output.append(x)

        if self.concat:
            x = torch.cat(output, dim=-1)

        return x