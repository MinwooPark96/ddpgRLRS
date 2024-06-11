import os
import numpy as np
import pandas as pd

def load_whole_dataset(DATA_DIR: str):
    print('Whole Data loading...')

    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]
    
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=object)
    movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로 - 안쓸거임
    movies_id_to_movies = {movie[0]: movie[0:] for movie in movies_df.values}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(DATA_DIR + '/user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(DATA_DIR + '/users_histroy_len.npy')

    return ratings_df, users_dict, users_history_lens, movies_id_to_movies

def load_interact_dataset(DATA_DIR: str):
    print('Interact Data loading...')

    ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ml_1m.inter'), sep=',', dtype=np.uint32)
    ratings_df.columns = ['UserID', 'MovieID', 'Rating']
    users_list = np.loadtxt(os.path.join(DATA_DIR, 'users.csv'), dtype=int)
    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'items.csv'), dtype=int)
    movies_df.columns = ['MovieID']

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_df.values}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(DATA_DIR + '/user_dict_new.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(DATA_DIR + '/users_histroy_len_new.npy')

    return ratings_df, users_dict, users_history_lens, movies_id_to_movies