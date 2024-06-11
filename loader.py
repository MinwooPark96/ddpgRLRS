import os
import numpy as np
import pandas as pd

def load_dataset(data_dir: str, mode: str):
    """
    DATA_DIR에 items.csv, final_user_dict.npy, final_users_history_len.npy 파일이 있어야 함

    return:
    - users_num : train(eval)에 사용될 user의 수
    - total_items_num : 전체 item의 수
    - users_dict : {user_id: [movie1, movie2, ...]} 형태의 dict
    - users_history_lens : 각 user의 history 길이(전체 user 대상)
    - movies_id_to_movies : {movie_id: movie_title} 형태의 dict -> 데이터 없음, 사용 안함
    """

    assert mode in ['train', 'eval'], "mode should be either 'train' or 'eval'"

    print('Interact Data loading...')

    movies_df = pd.read_csv(os.path.join(data_dir, 'items.csv'), dtype=int, header=None)
    movies_df.columns = ['MovieID']
    
    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_df.values}

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(data_dir + 'final_user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(data_dir + 'final_users_history_len.npy')

    total_users_num = len(users_dict.item()) 
    total_items_num = len(movies_df)
    print(f"total users_num : {total_users_num}, total items_num : {total_items_num}")

    train_users_num = int(total_users_num * 0.8)

    if mode == 'train':
        users_num = train_users_num
        users_dict = {k: users_dict.item().get(k) for k in range(users_num)}
        # users_history_lens = users_history_lens[:users_num]
        print(f"train_users_num : {users_num}")
    
    elif mode == 'eval':
        users_num = total_users_num - train_users_num 
        users_dict = {k: users_dict.item().get(k) for k in range(train_users_num, total_users_num)}
        # users_history_lens = users_history_lens[-users_num:]
        print(f"eval_users_num : {users_num}")
    else :
        raise ValueError("Invalid mode")
    
    print("Done")

    return users_num, total_items_num, users_dict, users_history_lens, movies_id_to_movies 

def load_dataset_session(DATA_DIR: str, mode: str="train"):
    """
    DATA_DIR에 items.csv, final_user_dict.npy, final_users_history_len.npy 파일이 있어야 함

    return:
    - users_num : 전체user의 수
    - total_items_num : 전체 item의 수
    - users_dict : {user_id: [movie1, movie2, ...]} 형태의 dict
    - users_history_lens : 각 user의 history 길이(전체 user 대상)
    - movies_id_to_movies : {movie_id: movie_title} 형태의 dict -> 데이터 없음, 사용 안함
    """

    assert mode in ['train', 'eval'], "mode should be either 'train' or 'eval'"

    print('Interact Data loading...')

    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'items.csv'), dtype=int, header=None)
    movies_df.columns = ['MovieID']
    
    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_df.values}

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(DATA_DIR + 'final_user_dict.npy', allow_pickle=True)

    total_users_num = len(users_dict.item()) 
    total_items_num = len(movies_df)
    print(f"total users_num : {total_users_num}, total items_num : {total_items_num}")

    # users_dict를 돌면서 history 앞에 20%는 eval, 나머지는 train으로 나누기
    train_users_dict = {}
    eval_users_dict = {}

    for userid, movie_ratings in users_dict.item().items():
        split_index = int(len(movie_ratings) * 0.2)
        eval_users_dict[userid] = movie_ratings[:split_index]
        train_users_dict[userid] = movie_ratings[split_index:]

    if mode == 'train':
        users_dict = train_users_dict
        users_history_lens = np.load(DATA_DIR + 'final_train_users_history_len.npy')
    
    else:
        users_dict = eval_users_dict
        users_history_lens = np.load(DATA_DIR + 'final_eval_users_history_len.npy')

    return total_users_num, total_items_num, users_dict, users_history_lens, movies_id_to_movies  

if __name__ == "__main__":
    ROOT_DIR = os.getcwd()
    data_dir = os.path.join(ROOT_DIR, 'data/ml-1m/')
    eval_users_num, _, test_users_dict, test_users_history_lens, _ = load_dataset(data_dir, 'test')