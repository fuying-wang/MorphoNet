import ipdb
import os
import pandas as pd
from datetime import datetime


result_dir = "/home/fywang/Documents/SPANTHER/src/results"
# dataset_name = "NSCLC"
dataset_name = "BRCA"
# dataset_name = "KIRC"
# dataset_name = "LUAD"
# dataset_name = "UCEC"
# dataset_name = "BLCA"
# dataset_name = "COADREAD"
task = "survival"  # survival or classification
# method = "OT"
# method = "GNN"
method = "TransMIL"
# method  = "ABMIL"
# method = "GraphTransformer"
# method = "DeepAttnMISL"
# method = "SPANTHER"
feat_dir = "superpatch_mean"
# feat_dir = "superpatch_connected_ot_C64_graph"
# feat_dir = "superpatch_connected"
# feat_dir = "superpatch_leiden_mean_graph"
# feat_dir = "superpatch_ot_graph"
# feat_dir = "WSI_graph"
# feat_dir = "superpatch_kmeans_mean_graph"
# feat_dir = "h5_files"
finetune_method = "LinearEmb" # only used for PANTHER or SPANTHER
if task == "survival":
    n_folds = 5
elif task == "classification":
    n_folds = 1


def main():
    result_folder = f"{dataset_name}_{task}::{method}_default::uni_pt_patch_features/{dataset_name}_{task}"
    result_path = os.path.join(result_dir, result_folder)
    assert os.path.exists(result_path), f"Result path {result_path} does not exist."
    all_df = []
    all_times = []
    for k in range(n_folds):
        fold_result_path = os.path.join(result_path, f"k={k}")
        assert os.path.exists(fold_result_path), f"Fold result path {fold_result_path} does not exist."
        if method in ["PANTHER", "SPANTHER"]:
            temp = f"TCGA_{dataset_name}_overall_survival::{method}_default+{finetune_method}::cox::{feat_dir}"
        else:
            temp = f"TCGA_{dataset_name}_overall_survival::{method}_default::{feat_dir}"
        fold_result_path = os.path.join(fold_result_path, temp)
        assert os.path.exists(fold_result_path), f"Fold result path {fold_result_path} does not exist."
        folder_names = sorted(os.listdir(fold_result_path))
        latest_folder = folder_names[-1]
        cur_time = latest_folder.split("::")[-1]
        all_times.append(cur_time)
        fold_result_path = os.path.join(fold_result_path, latest_folder)
        df = pd.read_csv(os.path.join(fold_result_path, "summary.csv"))
        all_df.append(df)
    all_df = pd.concat(all_df)
    all_df["time"] = all_times
    if task == "survival":
        if dataset_name == "LUAD":
            all_df["c_index_mean"] = all_df["c_index_test_tcgaluad"].mean()
            all_df["c_index_std"] = all_df["c_index_test_tcgaluad"].std()    
        elif dataset_name == "KIRC":
            all_df["c_index_mean"] = all_df["c_index_test"].mean()
            all_df["c_index_std"] = all_df["c_index_test"].std()  
            # all_df["c_index_cptac_mean"] = all_df["c_index_filtered_test_cptac"].mean()
            # all_df["c_index_cptac_std"] = all_df["c_index_filtered_test_cptac"].std()  
        else:
            all_df["c_index_mean"] = all_df["c_index_test"].mean()
            all_df["c_index_std"] = all_df["c_index_test"].std()
    elif task == "classification":
        raise NotImplementedError
        # all_df["accuracy_mean"] = all_df["accuracy"].mean()
        # all_df["accuracy_std"] = all_df["accuracy"].std()
    print(f"Dataset: {dataset_name}\tTask: {task}\tMethod: {method}\tFeature: {feat_dir}")
    print(all_df.T)

if __name__ == "__main__":
    main()