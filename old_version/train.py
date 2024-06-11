# Dependencies
import pandas as pd
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent

import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')

STATE_SIZE = 10
MAX_EPISODE_NUM = 50000

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

    print('Data loading...')

    # # 데이터 로딩
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]

    # # 데이터프레임 생성
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    ratings_df = ratings_df.astype({'UserID': np.int32, 'MovieID': np.int32, 'Rating': np.int32, 'Timestamp': np.int64})

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리된 파일 로드
    users_dict = np.load(ROOT_DIR + '/data/user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(ROOT_DIR + '/data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k: users_dict.item().get(k)
                        for k in range(1, train_users_num+1)}
    
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict,
                    train_users_history_lens,
                    movies_id_to_movies,
                    STATE_SIZE)
    
    recommender = DRRAgent(env,
                           users_num,
                           items_num,
                           STATE_SIZE,
                           use_wandb=False)
    
    recommender.train(MAX_EPISODE_NUM, load_model=False, top_k=5)