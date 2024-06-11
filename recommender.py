import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
import matplotlib.pyplot as plt
import os
import wandb
from replay_buffer import PriorityExperienceReplay
from actor import Actor
from critic import Critic
from embedding import UserMovieEmbedding, UserMovieMultiModalEmbedding
from state_representation import DRRAveStateRepresentation
import time

from typing import Sequence, Collection, AbstractSet, List, Optional, Tuple, Mapping, Callable, Dict, TypeVar, FrozenSet, Any, Union, Iterable

import tensorflow as tf

ROOT_DIR = os.getcwd()
SAVE_DIR = os.path.join(ROOT_DIR, 'save_model')

class DRRAgent:
    def __init__(self,
                 env ,
                 users_num: int,
                 items_num: int, 
                 state_size: int, 
                 is_eval: bool, 
                 use_wandb: bool,
                 embedding_dim: int,
                 actor_hidden_dim: int,
                 actor_learning_rate: float,
                 critic_hidden_dim: int,
                 critic_learning_rate: float,
                 discount: float,
                 tau: float,
                 memory_size: int,
                 batch_size: int,
                 epsilon: float,
                 std: float,
                 args
                 ):
        
        self.env = env
        self.args = args
        if args.gpu == -1:
            self.device = torch.device('cpu')
        else :
            self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        print(f'[DEVICE] Using {self.device}')
        
        self.users_num = users_num
        self.items_num = items_num
        self.state_size = state_size
        
        self.embedding_dim = embedding_dim        
        # actor network hyperparameters
        self.actor_hidden_dim = actor_hidden_dim
        self.actor_learning_rate = actor_learning_rate
        
        # critic network hyperparameters
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_learning_rate = critic_learning_rate
        
        self.discount_factor = discount
        self.tau = tau

        self.replay_memory_size = memory_size
        self.batch_size = batch_size

        self.actor = Actor(
            embedding_dim = self.embedding_dim,
            hidden_dim = self.actor_hidden_dim,
            learning_rate = self.actor_learning_rate,
            tau = self.tau)
        
        self.critic = Critic(
            embedding_dim = self.embedding_dim,
            hidden_dim = self.critic_hidden_dim, 
            learning_rate = self.critic_learning_rate,
            tau = self.tau)
        
    
        if not args.modality:
            self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
            embedding_save_file_dir = ROOT_DIR + '/save_weights/user_movie_embedding_case4.pth'
            embedding_network_checkpoint = torch.load(embedding_save_file_dir)
            self.embedding_network.m_embedding.weight.data = embedding_network_checkpoint['m_embedding.weight']
            self.embedding_network.u_embedding.weight.data = embedding_network_checkpoint['u_embedding.weight']
            self.embedding_network.eval()
            print('[Embedding] UserMovieEmbedding is loaded')
        
        else :
            modality = tuple(args.modality.lower().split(','))
            self.embedding_network = UserMovieMultiModalEmbedding(users_num,
                                                                  items_num,
                                                                  self.embedding_dim,
                                                                  modality,
                                                                  args.fusion,
                                                                  args.aggregation)
            
            self.embedding_network([np.array([0, 1]), np.array([0, 1])])
            
            mod_name = ''.join([mod[0] for mod in modality]).upper()
            weights_name = f'{mod_name}_{args.fusion}_{args.aggregation}'
            
            embedding_save_file_dir = os.path.join(ROOT_DIR, 'save_weights', f'u_m_model_{weights_name}.h5')
            self.embedding_network.load_weights(embedding_save_file_dir)
            print('[Embedding] UserMovieMultiModalEmbedding is loaded')
            
        time.sleep(2)
        
        assert os.path.exists(
            embedding_save_file_dir), f"embedding save file directory: '{embedding_save_file_dir}' is wrong."

        self.srm_ave = DRRAveStateRepresentation(embedding_dim = self.embedding_dim, state_size = self.state_size)
        self.srm_ave.eval()
        
        # PER
        self.buffer = PriorityExperienceReplay(
            self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # ε-greedy exploration hyperparameter
        self.epsilon = epsilon
        self.epsilon_decay = (self.epsilon - 0.1) / args.max_episode_num
        self.std = std

        self.is_eval = is_eval

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr",
                       entity="diominor",
                       config={'users_num': users_num,
                               'items_num': items_num,
                               'state_size': state_size,
                               'embedding_dim': self.embedding_dim,
                               'actor_hidden_dim': self.actor_hidden_dim,
                               'actor_learning_rate': self.actor_learning_rate,
                               'critic_hidden_dim': self.critic_hidden_dim,
                               'critic_learning_rate': self.critic_learning_rate,
                               'discount_factor': self.discount_factor,
                               'tau': self.tau,
                               'replay_memory_size': self.replay_memory_size,
                               'batch_size': self.batch_size,
                               'std_for_exploration': self.std})

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        
        for i in range(q_values.shape[0]):
            # y_t = r + discount_factor * Q(s)
            y_t[i] = rewards[i] + (1 - dones[i]) * (self.discount_factor * q_values[i])
        
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        
        if items_ids is None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))
        
        if not self.args.modality:
            items_ebs = self.embedding_network.m_embedding(torch.tensor(items_ids, dtype=torch.long)).to(self.device) 
        else :
             _, items_ebs = self.embedding_network.get_embedding([np.zeros_like(items_ids), np.array(items_ids)])
             items_ebs = torch.from_numpy(items_ebs.numpy()).to(self.device)
        
        action = torch.transpose(action, 0, 1)
        
        if top_k:
            item_indice = torch.argsort(torch.matmul(items_ebs, action).squeeze())[-top_k:]
            return items_ids[item_indice]
        
        else:
            item_idx = torch.argmax(torch.matmul(items_ebs, action)).item()
            return items_ids[item_idx]

    def train(self, max_episode_num: int,
              top_k: Optional[int] = False,
              load_model: Optional[int] = False):
        
        # weight save directory
        self.save_model_weight_dir = SAVE_DIR + f"/train_{self.args.modality}_{self.args.fusion}_{self.args.aggregation}_session"
        
        if not os.path.exists(self.save_model_weight_dir):
            os.makedirs(os.path.join(self.save_model_weight_dir, 'images'))
        
        
        # Initialize target networks
        self.actor.build_networks()
        self.critic.build_networks()
        
        # Move networks to device
        self.actor.network.to(self.device)
        self.critic.network.to(self.device)
        self.actor.target_network.to(self.device)
        self.critic.target_network.to(self.device)
        
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            self.load_model(ROOT_DIR + self.args.checkpoint,
                            ROOT_DIR + self.args.checkpoint)
            print('Completely load weights!')
            time.sleep(3)

        episodic_precision_history = []

        for episode in range(max_episode_num):
            
            # Reset episodic reward
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0
            
            # Reset environment
            user_id, items_ids, done = self.env.reset()

            while not done:
                
                if not self.args.modality:
                    # SINGLE
                    user_eb = self.embedding_network.u_embedding(torch.tensor([user_id], dtype=torch.long)).detach().numpy()
                    items_eb = self.embedding_network.m_embedding(torch.tensor(items_ids, dtype=torch.long)).detach().numpy()
                
                else :
                    # MULTIMODAL
                    tf_items_ids = tf.convert_to_tensor(items_ids, dtype=tf.int32)
                    tf_user_id = tf.convert_to_tensor(user_id, dtype=tf.int32)
                    user_eb, items_eb = self.embedding_network.get_embedding([tf_user_id, tf_items_ids])
                    user_eb, items_eb = tf.reshape(user_eb, (1,self.embedding_dim)).numpy(), items_eb.numpy()
                

                state = self.srm_ave([
                    torch.tensor(user_eb, dtype=torch.float32), 
                    torch.tensor(items_eb, dtype=torch.float32).unsqueeze(0)
                ]).to(self.device)
                
                # Get action (ranking score)
                action = self.actor.network(state)

                # ε-greedy exploration
                if self.epsilon > np.random.uniform() and not self.is_eval:
                    self.epsilon -= self.epsilon_decay
                    action += torch.tensor(np.random.normal(0, self.std, size=action.shape), dtype=torch.float32).to(self.device)

                # Recommend item
                recommended_item = self.recommend_item(
                    action, self.env.recommended_items, top_k=top_k)

                # Calculate reward & observe new state (in env)
                # Step
                next_items_ids, reward, done, _ = self.env.step(
                    recommended_item, top_k=top_k)
                if top_k:
                    reward = np.sum(reward)

                # Get next state
                if not self.args.modality:
                    next_items_eb = self.embedding_network.m_embedding(torch.tensor(next_items_ids, dtype=torch.long)).detach().numpy()
                else :
                    _, next_items_eb = self.embedding_network.get_embedding([np.zeros_like(next_items_ids), np.array(next_items_ids)])
                    next_items_eb = next_items_eb.numpy()
                
                next_state = self.srm_ave([
                    torch.tensor(user_eb, dtype=torch.float32), 
                    torch.tensor(next_items_eb, dtype=torch.float32).unsqueeze(0)
                ]).to(self.device)

                # Store in buffer
                action = action.detach().cpu().numpy()
                state = state.detach().cpu().numpy()
                next_state = next_state.detach().cpu().numpy()
                
                self.buffer.append(state, action, reward, next_state, done)

                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    # Sample a minibatch
                    batch = self.buffer.sample(self.batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = batch
                    
                    batch_states = batch_states.clone().to(self.device)
                    batch_actions = batch_actions.clone().to(self.device)
                    batch_rewards = batch_rewards.clone().to(self.device)
                    batch_next_states = batch_next_states.clone().to(self.device)
                    batch_dones = torch.tensor(batch_dones).to(self.device)
                    weight_batch = torch.tensor(weight_batch).to(self.device)
                    
                    # Set TD targets
                    target_next_action = self.actor.target_network(batch_next_states)
                    qs = self.critic.network([target_next_action, batch_next_states])
                    target_qs = self.critic.target_network([target_next_action, batch_next_states])
                    min_qs = torch.min(torch.cat([target_qs, qs], dim=1), dim=1, keepdim=True).values  # Double Q method
                    td_targets = self.calculate_td_target(batch_rewards.cpu().numpy(), min_qs.detach().cpu().numpy(), batch_dones.cpu().numpy())

                    td_targets = torch.tensor(td_targets).to(self.device)
                     
                    # Update priority
                    for (p, i) in zip(td_targets, index_batch):
                        self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    # Update critic network
                    q_loss += self.critic.train([batch_actions, batch_states], td_targets, weight_batch)

                    # Update actor network
                    s_grads = self.critic.dq_da([batch_actions, batch_states])
                    self.actor.train(batch_states, s_grads)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward
                mean_action += np.sum(action[0]).item() / len(action[0])
                steps += 1

                if reward > 0:
                    correct_count += 1

                print(
                    f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')

                if done:
                    print()
                    precision = int(correct_count / steps * 100)
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss/steps}, mean_action : {mean_action/steps}')
                    if self.use_wandb:
                        wandb.log({'precision': precision, 'total_reward': episode_reward,
                                  'epsilon': self.epsilon, 'q_loss': q_loss / steps, 'mean_action': mean_action / steps})
                    episodic_precision_history.append(precision)

            if (episode + 1) % 50 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig(os.path.join(self.save_model_weight_dir, 'images/training_precision_%_top_5.png'))

            if (episode + 1) % 1000 == 0 or episode == max_episode_num - 1:
                self.save_model(os.path.join(self.save_model_weight_dir, f'actor_{episode + 1}_fixed.pth'),
                                os.path.join(self.save_model_weight_dir, f'critic_{episode + 1}_fixed.pth'))

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.network.state_dict(), actor_path)
        torch.save(self.critic.network.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        self.actor.network.load_state_dict(torch.load(actor_path))
        self.critic.network.load_state_dict(torch.load(critic_path))
        
    def eval(self):
        self.actor.network.eval()
        self.critic.network.eval()
        