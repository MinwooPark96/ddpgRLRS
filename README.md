# Deep Reinforcement Learning based Recommender System in Torch
The implemetation of Deep Reinforcement Learning based Recommender System from the paper [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/abs/1810.12027) by Liu et al. Build recommender system with [DDPG](https://arxiv.org/abs/1509.02971) algorithm. Add state representation module to produce trainable state for RL algorithm from data.

# Dataset
[MovieLens 1M Datset](https://grouplens.org/datasets/movielens/1m/)

```
unzip ./ml-1m.zip
```

# Embedding 
We incorporate multimodal movie features, utilizing audio, text, and video data, to create a more comprehensive and context-aware recommendation model. Processed by modality-specific encoders including ViT (visual), AST (audio), and BERT (text).

- Early Fusion : One FC layer applied to pooled multi-modal features.
- Late Fusion : Modality-specific FC layers applied before pooling.
- Pooling : concatenation / element-wise mean

# Environment Setting
```
conda create -n env_name python=3.11.2 

conda activate env_name

conda install tensorflow==2.12.0
pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

# run
1. run DRR with multi modal feature (dimension of embedding = 128(64*2))
    ```
    bash scripts/train_modality.sh
    bash scripts/eval_modality.sh
    ```
2. run DRR with single ID feature (dimension of embedding = 100)
    ```
    bash scripts/train.sh
    bash scripts/eval.sh
    ```


# reference
https://github.com/backgom2357/Recommender_system_via_deep_RL