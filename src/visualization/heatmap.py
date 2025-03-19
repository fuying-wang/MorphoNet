import cv2
import h5py
import pandas as pd
import scanpy as sc 
import os
import ipdb
import openslide
import numpy as np
from tqdm import tqdm
import torch
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import networkx as nx
import scanpy as sc
import spatialleiden as sl
import squidpy as sq
from sklearn.cluster import KMeans
import sys
sys.path.append('../../src/')
from mil_models import GNN, GNNConfig
from prototype_visualization_utils import get_panther_encoder, visualize_categorical_heatmap, get_mixture_plot, get_default_cmap
from WholeSlideImage import WholeSlideImage


# slide_id = "TCGA-69-7760-01Z-00-DX1.7fc295e3-5bfc-4017-801c-491489d0eb34" # LUAD
slide_id = "TCGA-21-A5DI-01Z-00-DX1.E9123261-ADE7-468C-9E9A-334E131FFF97" # LUSC
wsi_dir = "/data1/r20user2/wsi_data/TCGA_NSCLC/WSIs"
wsi_path = os.path.join(wsi_dir, f"{slide_id}.svs")
graph_dir = "/data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_graph"
graph_path = os.path.join(graph_dir, f"{slide_id}.pt")
h5_dir = "/data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/h5_files"
with h5py.File(os.path.join(h5_dir, f"{slide_id}.h5"), "r") as f:
    print(list(f.keys()))
    coords = f["coords"][:]
    features = f["features"][:]
pt_dir = "/data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/pt_files"
pt_path = os.path.join(pt_dir, f"{slide_id}.pt")
pt_data = torch.load(pt_path)
graph_data = torch.load(graph_path)
slide_prototype_dir = "/data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/cluster_feats"
slide_prototype_path = os.path.join(slide_prototype_dir, f"{slide_id}.h5ad")
slide_prototype = sc.read_h5ad(slide_prototype_path)

# connected_dir = "/data1/r20user2/wsi_data/TCGA_NSCLC/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_connected_ot"
# with h5py.File(os.path.join(connected_dir, f"{slide_id}.h5"), "r") as f:
#     print(list(f.keys()))
#     connected_coords = f["coords"][:]
#     connected_features = f["features"][:]

#
config = GNNConfig()
config.in_dim = 1024
model = GNN(config, mode="classification")
ckpt = "/home/fywang/Documents/SPANTHER/src/results/NSCLC_classification::GNN_default::uni_pt_patch_features/NSCLC_classification/k=0/NSCLC::GNN_default::superpatch_graph/NSCLC::GNN_default::superpatch_graph::25-03-08-21-44-56/s_checkpoint.pth"
model.load_state_dict(torch.load(ckpt)["model"])

#
x, edge_index = graph_data.x, graph_data.edge_index
x = model.preprocess(x)
if model.num_layers > 0:
    for conv in model.convs_list:
        x = conv(x, edge_index)
else:
    # in this case, it is identity
    x = model.convs_list(x)

A, x = model.attention_net(x)
A = torch.transpose(A, 0, 1)
A = F.softmax(A, dim=1)
x = torch.mm(A, x)

logits = model.fc(x)
print(logits)


# 
# # plot topk subbgraphs and whole WSI
# wsi = openslide.open_slide(wsi_path)
# downscale = 8
# vis_level = wsi.get_best_level_for_downsample(downscale)
# print(f"vis_level: {vis_level}")
# w, h = wsi.level_dimensions[vis_level]
# print(w, h)
# downsamples = wsi.level_downsamples[vis_level]
# print(downsamples)

# patch_size = 512
# vis_patch_size = int(patch_size / downsamples)
# whole_wsi = np.array(wsi.read_region((0, 0), level=vis_level, size=(w, h)).convert("RGB"))
# plt.imshow(whole_wsi)
# plt.axis("off")
# # plt.savefig(f"WSI_{slide_id}.pdf", bbox_inches="tight")

#
seed = 42

adata = sc.AnnData(features)
adata.obsm["spatial"] = coords

sc.pp.pca(adata, n_comps=64, random_state=seed)
sc.pp.neighbors(adata, random_state=seed)
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=8)
adata.obsp["spatial_connectivities"] = sl.distance2connectivity(
    adata.obsp["spatial_distances"]
)
sc.tl.leiden(adata, directed=False, random_state=seed)
sl.spatialleiden(adata, layer_ratio=1.8, directed=(False, True), seed=seed)

# # %%
# ### Visualize the categorical heatmap and the GMM mixtures
# cat_map = visualize_categorical_heatmap(
#     wsi,
#     coords, 
#     adata.obs["spatialleiden"].to_numpy().astype(int), 
#     label2color_dict=get_default_cmap(32),
#     vis_level=wsi.get_best_level_for_downsample(32),
#     patch_size=(patch_size, patch_size),
#     alpha=0.4,
# )

# resized_cat_map = cat_map.resize((cat_map.width, cat_map.height))
# # resized_cat_map.save(f"Spatial_Leiden_cluster_{slide_id}.pdf")
# display(resized_cat_map)


#
cluster_labels = adata.obs["spatialleiden"].to_numpy().astype(int)
heatmap = np.zeros(len(cluster_labels))
for i in np.unique(cluster_labels):
    # print(i)
    indices = np.where(cluster_labels == i)[0]
    # print(indices.shape)
    # print(heatmap[indices])
    heatmap[indices] = A[0][i].item()
print(heatmap)

wsi_object = WholeSlideImage(wsi_path)


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

heatmap = drawHeatmap(heatmap, coords, wsi_path, wsi_object=wsi_object,  
                    cmap="jet", alpha=0.4, 
                    binarize=False, 
                    blank_canvas=False,
                    segment=False,
                    thresh=-1,  
                    patch_size = 512)
        





