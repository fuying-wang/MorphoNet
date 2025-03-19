# MorphoNet

### Installation

```
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
conda create -n spanther python=3.10
pip install spatialleiden anndata scanpy
pip install squidpy
pip3 install torch torchvision torchaudio
```

### Data Structure

```
TCGA-RCC 
- extracted_mag20x_patch256
-- uni_pt_patch_features
--- superpatch_mean / superpatch_graph
```