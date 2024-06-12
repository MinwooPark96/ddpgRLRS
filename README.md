# Deep Reinforcement Learning based Recommender System in Torch

The implemetation of Deep Reinforcement Learning based Recommender System from the paper [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/abs/1810.12027) by Liu et al. Build recommender system with [DDPG](https://arxiv.org/abs/1509.02971) algorithm. Add state representation module to produce trainable state for RL algorithm from data.

# Dataset
[MovieLens 1M Datset](https://grouplens.org/datasets/movielens/1m/)

```
unzip ./ml-1m.zip
```

# Embedding 
The Embedding vector for video, audio and text is used. 

# Environment
```
conda create -n env_name python=3.8.15

conda activate env_name

# tensorflow
conda install -c conda-forge tensorflow=2.6.0=cuda112py38hbe5352d_2 pandas=1.4.4=py38h47df419_0 scikit-learn=0.23.2=py38h5d63f67_3 matplotlib=3.3.3=py38h578d9bd_0 wandb=0.13.5=pyhd8ed1ab_0 tensorflow-gpu

# torch
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

# reference

https://github.com/backgom2357/Recommender_system_via_deep_RL