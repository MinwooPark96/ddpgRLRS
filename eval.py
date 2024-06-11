
import numpy as np
import time
import os
import torch
import torch.nn.functional as F
import pandas as pd
from envs import OfflineEnv
from recommender import DRRAgent
from loader import load_dataset, load_dataset_session
import tensorflow as tf

"""
[Evaluation 방식 - Offline Evaluation (Algorithm 2)]
- eval_user_list에서 한명씩 평가진행
- 각 time step마다, 학습된 policy로 action 취하고 item 추천 -> reward 관찰, state update되고 추천된 item은 추천가능 목록에서 제거
- 한 user 당 몇번의 추천을 진행할지는 결정해야 할 듯 (Jupyter notebook에서는 한번만 하는 것으로 보이는데 알고리즘 상에는 T번해서 평균 내는 듯)
"""

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')

def evaluate(recommender, env, args, check_movies: bool=False, top_k: int=1, length: int=1):
    mean_precision = 0
    mean_ndcg = 0

    episode_reward = 0
    steps = 0

    user_id, items_ids, done = env.reset()
    print(f"[STARTING RECOMMENDATION TO USER {user_id}]")
    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')

    while not done:
        
        if not args.modality:
            user_eb = recommender.embedding_network.u_embedding(torch.tensor([user_id], dtype=torch.long))
            items_eb = recommender.embedding_network.m_embedding(torch.tensor(items_ids, dtype=torch.long))
        else :
            tf_items_ids = tf.convert_to_tensor(items_ids, dtype=tf.int32)
            tf_user_id = tf.convert_to_tensor(user_id, dtype=tf.int32)
            user_eb, items_eb = recommender.embedding_network.get_embedding([tf_user_id, tf_items_ids])
            user_eb, items_eb = tf.reshape(user_eb, (1,args.embedding_dim)).numpy(), items_eb.numpy()
            
        # state = recommender.srm_ave(user_eb.unsqueeze(0), items_eb.unsqueeze(0))
        state = recommender.srm_ave([
            torch.tensor(user_eb, dtype=torch.float32),
            torch.tensor(items_eb, dtype=torch.float32).unsqueeze(0)
            ])

        action = recommender.actor.network(state)

        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)

        next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)

        if check_movies:
            print(f'\t[step: {steps+1}] recommended items ids : {recommended_item}, reward : {reward}')

        correct_list = [1 if r > 0 else 0 for r in reward]

        dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])        
        mean_ndcg += dcg/idcg

        correct_num = len(reward) - correct_list.count(0)
        mean_precision += correct_num / len(reward)

        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1

        if done or steps >= length:
            break

    if check_movies:
        print(f"\tprecision@{top_k} : {mean_precision/steps}, ndcg@{top_k} : {mean_ndcg/steps}, episode_reward : {episode_reward/steps}\n")

    return mean_precision/steps, mean_ndcg/steps, episode_reward/steps

def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r)/np.log2(i+2)
        idcg += (ir)/np.log2(i+2)
    return dcg, idcg

def evaluater(args):
    
    total_users_num, total_items_num, eval_users_dict, users_history_lens, movies_id_to_movies = load_dataset_session(DATA_DIR, 'eval')

    sum_precision, sum_ndcg = 0, 0

    end_evaluation = 200

    temp_env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, args.state_size)
    avaiable_users = temp_env.available_users
    print(f"Available number of users: {len(avaiable_users)}")

    for i, user_id in enumerate(avaiable_users):
        env = OfflineEnv(eval_users_dict, 
                         users_history_lens, 
                         movies_id_to_movies, 
                         args.state_size, 
                         fix_user_id=user_id)
        
            
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
        
        recommender.eval()
        recommender.load_model(args.saved_actor, args.saved_critic)
        
        precision, ndcg, _ = evaluate(
            recommender,
            env,
            args,
            check_movies=True,
            top_k=args.top_k,
            length=args.state_size)
        
        sum_precision += precision
        sum_ndcg += ndcg

        if i > end_evaluation:
            break

    print("\n[FINAL RESULT]")
    print(f'precision@{args.top_k} : {sum_precision/(end_evaluation)}, ndcg@{args.top_k} : {sum_ndcg/(end_evaluation)}')