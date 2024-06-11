# Dependencies
import pandas as pd
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent
from loader import load_dataset, load_dataset_session
import os

"""
[Training 방식]
- 매 에피소드마다 user를 랜덤하게 선택 (OfflineEnv.user)
- user의 최근에 본 영화 10개를 이용해 state 생성
- user가 최근에 본 영화 10개를 제외한 나머지 영화들을 추천 (DRRAgent.recommend_item())
- 한 user당 trajectory 최대 길이는 약 3000, 그 전에 user가 본 영화 history 길이만큼 추천받으면 종료(OfflineEnv.step()) 
- Actor, Critic 파라미터 업데이트는 replay buffer에서 32개씩 batch로 묶어서 진행
"""

def trainer(args):
    
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
    

    total_users_num, total_items_num, train_users_dict, users_history_lens, movies_id_to_movies = load_dataset_session(DATA_DIR, 'train')

    
    time.sleep(2)
    
    env = OfflineEnv(train_users_dict,
                    users_history_lens,
                    movies_id_to_movies,
                    args.state_size)

    
    recommender = DRRAgent(env = env,
                            users_num = total_users_num,
                            items_num = total_items_num,
                            state_size = args.state_size,
                            is_eval = args.mode == 'eval',
                            use_wandb = args.use_wandb,
                            embedding_dim = args.dim_emb,
                            actor_hidden_dim = args.dim_actor,
                            actor_learning_rate = args.lr_actor,
                            critic_hidden_dim = args.dim_critic,
                            critic_learning_rate = args.lr_critic,
                            discount = args.discount,
                            tau = args.tau,
                            memory_size = args.memory_size,
                            batch_size = args.batch_size,
                            epsilon = args.epsilon,
                            std = args.std,
                            args = args,
                           )
    
    recommender.train(max_episode_num = args.max_episode_num, 
                      load_model = args.checkpoint, 
                      top_k=args.top_k)

# if __name__ == '__main__':
#     trainer()