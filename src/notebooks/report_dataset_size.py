import os
import numpy as np
import pandas as pd
import torch
import ipdb
from tqdm import tqdm


def main():
    split_dir = "/home/fywang/Documents/SPANTHER/src/dataset_csv"
    for dataset_name in ["TCGA_KIRC", "TCGA_LUAD"]:
        if dataset_name == "TCGA_KIRC":
            dataset_dir = os.path.join("/data1/r20user2/wsi_data", "TCGA_RCC")
        elif dataset_name == "TCGA_LUAD":
            dataset_dir = os.path.join("/data1/r20user2/wsi_data", "TCGA_NSCLC")
        else:
            dataset_dir = os.path.join("/data1/r20user2/wsi_data", dataset_name)
        superpatch_graph_dir = os.path.join(dataset_dir, "extracted_mag20x_patch256/uni_pt_patch_features/superpatch_graph")
        wsi_graph_dir = os.path.join(dataset_dir, "extracted_mag20x_patch256/uni_pt_patch_features/WSI_graph")
        
        if dataset_name == "TCGA_NSCLC":
            csv_path = os.path.join(split_dir, "NSCLC/classification.csv")
        elif dataset_name == "EBRAINS":
            csv_path = os.path.join(split_dir, "ebrains/classification.csv")
        elif dataset_name == "PANDA":
            csv_path = os.path.join(split_dir, "panda_wholesight/classification.csv")
        elif dataset_name == "TCGA_BRCA":
            csv_path = os.path.join(split_dir, "TCGA_BRCA/survival.csv")
        elif dataset_name == "TCGA_KIRC":
            csv_path = os.path.join(split_dir, "TCGA_KIRC/survival.csv")
        elif dataset_name == "TCGA_LUAD":
            csv_path = os.path.join(split_dir, "TCGA_LUAD/survival.csv")
        df = pd.read_csv(csv_path)

        superpatch_graph_node_list = []
        superpatch_graph_edge_list = [] 
        wsi_graph_node_list = []
        wsi_graph_edge_list = []
        for row in tqdm(df.itertuples(), total=len(df)):
            slide_id = row.slide_id
            superpatch_graph_path = os.path.join(superpatch_graph_dir, f"{slide_id}.pt")
            wsi_graph_path = os.path.join(wsi_graph_dir, f"{slide_id}.pt")
            superpatch_graph = torch.load(superpatch_graph_path)
            wsi_graph = torch.load(wsi_graph_path)
            superpatch_graph_node_list.append(superpatch_graph.num_nodes)
            superpatch_graph_edge_list.append(superpatch_graph.num_edges)
            wsi_graph_node_list.append(wsi_graph.num_nodes)
            wsi_graph_edge_list.append(wsi_graph.num_edges)

        print(f"Dataset: {dataset_name}")
        print(f"Superpatch graph node: {np.mean(superpatch_graph_node_list)}")
        print(f"Superpatch graph edge: {np.mean(superpatch_graph_edge_list)}")
        print(f"WSI graph node: {np.mean(wsi_graph_node_list)}")
        print(f"WSI graph edge: {np.mean(wsi_graph_edge_list)}")


def check_brca():
    dataset_dir = "/data1/r20user2/SPANTHER"
    for k in range(5):
        split_dir = os.path.join(dataset_dir, 
                                f"TCGA_BRCA_overall_survival_k={k}/extracted_mag20x_patch256/uni_pt_patch_features/superpatch_connected_panther_C64_graph")
        assert os.path.exists(split_dir)

        all_files = os.listdir(split_dir)
        all_files = [file for file in all_files if file.endswith(".pt")]
        for file in tqdm(all_files, total=len(all_files)):
            pass
            # if file.endswith(".pt"):
            #     try:
            #         graph = torch.load(os.path.join(split_dir, file))               
            #     except Exception as e:
            #         print(f"Error in {file}")                                                    



if __name__ == "__main__":
    # main()
    check_brca()