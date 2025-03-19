import argparse
import os
from collections import defaultdict

import h5py
import ipdb
import numpy as np
import openslide as osd
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms import Polar
import networkx as nx
import torch_geometric.utils as g_util
from tqdm import tqdm


def supernode_generation_per_slide(h5_file, feature_dir, save_dir, imagesize=256, spatial_threshold_1=2.9,
                                   spatial_threshold_2=5.5, feature_threshold=0.75, min_node_size=100):

    if os.path.exists(os.path.join(save_dir, h5_file[:-2] + "pt")):
        return

    h5_full_path = os.path.join(feature_dir, h5_file)

    # read file from h5py
    try:
        with h5py.File(h5_full_path, "r") as f:
            coords = f["coords"][()]
            features = f["features"][()]
    except:
        print(h5_full_path)

    feature_df = pd.DataFrame.from_dict(features)
    coordinate_df = pd.DataFrame.from_dict(coords / imagesize)
    coordinate_df.columns = ["X", "Y"]

    graph_dataframe = pd.concat([coordinate_df, feature_df], axis=1)
    graph_dataframe.sort_values(by=['Y', 'X'], inplace=True)
    graph_dataframe.reset_index(drop=True, inplace=True)

    max_X = graph_dataframe["X"].max()
    min_X = graph_dataframe["X"].min()
    max_Y = graph_dataframe["Y"].max()
    min_Y = graph_dataframe["Y"].min()

    # split the WSI into 6x6 regions
    gridNum = 6
    X_size = int((max_X - min_X) / gridNum)  # 291
    Y_size = int((max_Y - min_Y) / gridNum)  # 496

    node_dict = defaultdict(set)
    for p in range(gridNum+2):
        for q in range(gridNum+2):
            if p == 0:
                if q == 0:
                    is_X = graph_dataframe['X'] <= X_size * (p+1) + min_X
                    is_X2 = graph_dataframe['X'] >= min_X
                    is_Y = graph_dataframe['Y'] <= Y_size * (q+1) + min_Y
                    is_Y2 = graph_dataframe['Y'] >= min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                elif q == (gridNum+1):
                    is_X = graph_dataframe['X'] <= X_size * (p+1) + min_X
                    is_X2 = graph_dataframe['X'] >= min_X
                    is_Y = graph_dataframe['Y'] <= max_Y
                    is_Y2 = graph_dataframe['Y'] >= (
                        Y_size * (q) - 2) + min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                else:
                    is_X = graph_dataframe['X'] <= X_size * (p+1) + min_X
                    is_X2 = graph_dataframe['X'] >= min_X
                    is_Y = graph_dataframe['Y'] <= Y_size * (q+1) + min_Y
                    is_Y2 = graph_dataframe['Y'] >= (
                        Y_size * (q) - 2) + min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
            elif p == (gridNum+1):
                if q == 0:
                    is_X = graph_dataframe['X'] <= max_X
                    is_X2 = graph_dataframe['X'] >= (
                        X_size * (p) - 2) + min_X
                    is_Y = graph_dataframe['Y'] <= Y_size * (q+1) + min_X
                    is_Y2 = graph_dataframe['Y'] >= min_X
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                elif q == (gridNum+1):
                    is_X = graph_dataframe['X'] <= max_X
                    is_X2 = graph_dataframe['X'] >= (
                        X_size * (p) - 2) + min_X
                    is_Y = graph_dataframe['Y'] <= max_Y
                    is_Y2 = graph_dataframe['Y'] >= (
                        Y_size * (q) - 2) + min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                else:
                    is_X = graph_dataframe['X'] <= max_X
                    is_X2 = graph_dataframe['X'] >= (
                        X_size * (p) - 2) + min_X
                    is_Y = graph_dataframe['Y'] <= Y_size * (q+1) + min_Y
                    is_Y2 = graph_dataframe['Y'] >= (
                        Y_size * (q) - 2) + min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
            else:
                if q == 0:
                    is_X = graph_dataframe['X'] <= X_size * (p+1) + min_X
                    is_X2 = graph_dataframe['X'] >= (
                        X_size * (p) - 2) + min_X
                    is_Y = graph_dataframe['Y'] <= Y_size * (q+1) + min_Y
                    is_Y2 = graph_dataframe['Y'] >= min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                elif q == (gridNum+1):
                    is_X = graph_dataframe['X'] <= X_size * (p+1) + min_X
                    is_X2 = graph_dataframe['X'] >= (
                        X_size * (p) - 2) + min_X
                    is_Y = graph_dataframe['Y'] <= max_Y
                    is_Y2 = graph_dataframe['Y'] >= (
                        Y_size * (q) - 2) + min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                else:
                    is_X = graph_dataframe['X'] <= X_size * (p+1) + min_X
                    is_X2 = graph_dataframe['X'] >= (
                        X_size * (p) - 2) + min_X
                    is_Y = graph_dataframe['Y'] <= Y_size * (q+1) + min_Y
                    is_Y2 = graph_dataframe['Y'] >= (
                        Y_size * (q) - 2) + min_Y
                    X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

            if len(X_10) == 0:
                continue

            index_list = X_10.index.tolist()
            coordinate_dataframe = X_10.iloc[:, :2]
            coordinate_matrix = euclidean_distances(
                coordinate_dataframe.values, coordinate_dataframe.values)
            coordinate_matrix = np.where(
                coordinate_matrix > spatial_threshold_1, 0, 1)

            feature_dataframe = X_10.iloc[:, 2:]
            cosine_matrix = cosine_similarity(
                feature_dataframe.values, feature_dataframe.values)

            # Two conditions:
            # 1. spatial distance is smaller than 2.9
            # 2. feature similarity is larger than 0.75
            Adj_list = (coordinate_matrix == 1).astype(
                int) * (cosine_matrix >= feature_threshold).astype(int)

            for c, item in enumerate(Adj_list):
                for node_index in np.array(index_list)[item.astype('bool')]:
                    if node_index == index_list[c]:
                        pass
                    else:
                        # index_list[c]: the index of current node
                        # node_index: its adjacent
                        node_dict[index_list[c]].add(node_index)

    # compute the number of neighbors
    dict_len_list = []
    for i in range(0, len(node_dict)):
        dict_len_list.append(len(node_dict[i]))

    # now we sort the number of neighbors in the descending order
    arglist_strict = np.argsort(np.array(dict_len_list))[::-1]

    for arg_value in arglist_strict:
        if arg_value in node_dict.keys():
            for adj_item in node_dict[arg_value]:
                if adj_item in node_dict.keys():
                    node_dict.pop(adj_item)
                    arglist_strict = np.delete(
                        arglist_strict, np.argwhere(arglist_strict == adj_item))

    # coordinate x, coordinate y, features
    supernode_coordinate_x = []
    supernode_coordinate_y = []
    supernode_features = []
    whole_feature = graph_dataframe[graph_dataframe.columns.difference([
                                                                       'X', 'Y'])]

    for key_value in node_dict.keys():
        supernode_coordinate_x.append(graph_dataframe['X'][key_value])
        supernode_coordinate_y.append(graph_dataframe['Y'][key_value])
        if len(node_dict[key_value]) == 0:
            select_feature = whole_feature.loc[key_value].values
        else:
            # we use the average features of neighbors to represent this superpatch
            select_feature = whole_feature.loc[list(
                node_dict[key_value]) + [key_value]].values
            select_feature = select_feature.mean(axis=0)

        supernode_features.append(select_feature.reshape(1, -1))

    supernode_features = np.vstack(supernode_features)

    coordinate_integrate = pd.DataFrame(
        {'X': supernode_coordinate_x, 'Y': supernode_coordinate_y})
    coordinate_matrix1 = euclidean_distances(
        coordinate_integrate, coordinate_integrate)
    coordinate_matrix1 = np.where(
        coordinate_matrix1 > spatial_threshold_2, 0, 1)

    fromlist = []
    tolist = []

    for i in range(len(coordinate_matrix1)):
        temp = coordinate_matrix1[i, :]
        selectindex = np.where(temp > 0)[0].tolist()
        for index in selectindex:
            fromlist.append(int(i))
            tolist.append(int(index))

    edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
    x = torch.tensor(supernode_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    node_dict = pd.DataFrame.from_dict(node_dict, orient='index')

    # false graph filtering
    distance_thresh = spatial_threshold_2
    supernode_num = node_dict.index.tolist()
    coordinate_df = coordinate_df.loc[supernode_num, :]

    coordinate_matrix = pairwise_distances(coordinate_df.values, n_jobs=8)
    adj_matrix = np.where(coordinate_matrix >= distance_thresh, 0, 1)
    Edge_label = np.where(adj_matrix == 1)

    Adj_from = np.unique(Edge_label[0], return_counts=True)
    Adj_to = np.unique(Edge_label[1], return_counts=True)
    # search these nodes which only have one xxx
    Adj_from_singleton = Adj_from[0][Adj_from[1] == 1]
    Adj_to_singleton = Adj_to[0][Adj_to[1] == 1]
    Adj_singleton = np.intersect1d(Adj_from_singleton, Adj_to_singleton)
    coordinate_matrix_modify = coordinate_matrix

    fromlist = Edge_label[0].tolist()
    tolist = Edge_label[1].tolist()

    edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
    data.edge_index = edge_index

    connected_graph = g_util.to_networkx(data, to_undirected=True)
    # we extract subgraphs with node size >= min_node_size
    connected_graph = [connected_graph.subgraph(item_graph).copy() for item_graph in
                       nx.connected_components(connected_graph) if len(item_graph) > min_node_size]

    if len(connected_graph) == 0:
        return

    connected_graph_node_list = []
    for graph_item in connected_graph:
        connected_graph_node_list.extend(list(graph_item.nodes))
    # connected_graph = connected_graph_node_list
    # connected_graph = list(connected_graph)
    new_node_order_dict = dict(
        zip(connected_graph_node_list, range(len(connected_graph_node_list))))

    new_feature = data.x[connected_graph_node_list]
    new_edge_index = data.edge_index.numpy()
    new_edge_mask_from = np.isin(new_edge_index[0], connected_graph_node_list)
    new_edge_mask_to = np.isin(new_edge_index[1], connected_graph_node_list)
    new_edge_mask = new_edge_mask_from * new_edge_mask_to
    new_edge_index_from = new_edge_index[0]
    new_edge_index_from = new_edge_index_from[new_edge_mask]
    new_edge_index_from = [new_node_order_dict[item]
                           for item in new_edge_index_from]
    new_edge_index_to = new_edge_index[1]
    new_edge_index_to = new_edge_index_to[new_edge_mask]
    new_edge_index_to = [new_node_order_dict[item]
                         for item in new_edge_index_to]

    new_edge_index = torch.tensor(
        [new_edge_index_from, new_edge_index_to], dtype=torch.long)

    # new_supernode = node_dict.iloc[connected_graph_node_list, :]

    actual_pos = coordinate_df.iloc[connected_graph_node_list, :].values
    actual_pos = torch.tensor(actual_pos).float()
    # Saves the polar coordinates of linked nodes in its edge attributes
    pos_transfrom = Polar()
    new_graph = Data(x=new_feature, edge_index=new_edge_index,
                     pos=actual_pos * imagesize)
    try:
        new_graph = pos_transfrom(new_graph)
    except:
        ipdb.set_trace()

    transfer = T.ToSparseTensor()
    data = transfer(new_graph)
    torch.save(data, os.path.join(save_dir, h5_file[:-2] + "pt"))

    # for i in range(len(data.pos)):
    #     cond = (data.pos.numpy()[i, 0] == coords[:, 0]) & (
    #         data.pos.numpy()[i, 1] == coords[:, 1])
    #     assert cond.sum() == 1, f"{data.pos[i]}"

    return data


'''
python supernode_generation.py --data_source /data1/r20user2/wsi_data \
    --dataset_name TCGA_NSCLC 
'''

def main():
    parser = argparse.ArgumentParser(description="Supernode generation")
    parser.add_argument("--data_source", type=str, default="/data1/r20user2/wsi_data")
    parser.add_argument("--dataset_name", type=str, default="TCGA_RCC")
    parser.add_argument("--magnitude", type=int, default=20)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--spatial_threshold_1", type=float, default=1.9)
    parser.add_argument("--spatial_threshold_2", type=float, default=3.5)
    parser.add_argument("--feature_threshold", type=float, default=0.8)
    parser.add_argument("--min_node_size", type=int, default=25)
    parser.add_argument("--feature_dir", type=str,
                        default="extracted_mag20x_patch256/uni_pt_patch_features/h5_files")
    parser.add_argument("--save_dir", type=str,
                        default="extracted_mag20x_patch256/uni_pt_patch_features")
    args = parser.parse_args()

    args.imagesize = args.imagesize * (2 ** (40 / args.magnitude - 1))

    args.feature_dir = os.path.join(args.data_source, args.dataset_name, args.feature_dir)
    args.save_dir = os.path.join(args.data_source, args.dataset_name, args.save_dir, f"TEA_graph_{args.min_node_size}")

    # if args.dataset.lower() == "camelyon16":
    #     args.feature_dir = os.path.join(CAMELYON16_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         CAMELYON16_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_nsclc":
    #     args.feature_dir = os.path.join(NSCLC_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         NSCLC_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_rcc":
    #     args.feature_dir = os.path.join(RCC_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         RCC_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_brca":
    #     args.feature_dir = os.path.join(BRCA_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         BRCA_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_blca":
    #     args.feature_dir = os.path.join(BLCA_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         BLCA_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_ucec":
    #     args.feature_dir = os.path.join(UCEC_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         UCEC_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_esca":
    #     args.feature_dir = os.path.join(ESCA_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         ESCA_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # elif args.dataset.lower() == "tcga_prad":
    #     args.feature_dir = os.path.join(PRAD_DATA_DIR, args.feature_dir)
    #     args.save_dir = os.path.join(
    #         PRAD_DATA_DIR, args.save_dir, f"TEA_graph_{args.min_node_size}")
    # else:
    #     raise RuntimeError(f"No dataset {args.dataset}!")

    os.makedirs(args.save_dir, exist_ok=True)

    h5_files = sorted(os.listdir(args.feature_dir))

    for _, h5_file in tqdm(enumerate(h5_files), total=len(h5_files)):
        supernode_generation_per_slide(h5_file, args.feature_dir, args.save_dir, args.imagesize,
                                       args.spatial_threshold_1, args.spatial_threshold_2,
                                       args.feature_threshold, args.min_node_size)


if __name__ == "__main__":
    main()
