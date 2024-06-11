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
from embedding import UserMovieEmbedding
from state_representation import DRRAveStateRepresentation
from time import time

ROOT_DIR = os.getcwd()

class DRRAgent:
    def __init__(self,
                 env,
                 users_num,
                 items_num, 
                 state_size, 
                 is_test=False, 
                 use_wandb=False):
        
        self.env = env

        self.users_num = users_num
        self.items_num = items_num

        self.embedding_dim = 100
        
        # actor network hyperparameters
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        
        # critic network hyperparameters
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32

        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim,
                           self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(
            self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)

        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        
        self.embedding_network.eval() # MINWOO : freeze 하기 위해서 추가함.
        
        with torch.no_grad():
            self.embedding_network(torch.zeros((1, 2), dtype=torch.long))  # Example input to initialize

        # weight save directory
        self.save_model_weight_dir = ROOT_DIR + f"/save_model/trail-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        if not os.path.exists(self.save_model_weight_dir):
            os.makedirs(os.path.join(self.save_model_weight_dir, 'images'))
        
        embedding_save_file_dir = ROOT_DIR + '/save_weights/user_movie_embedding_case4.pth'
        
        assert os.path.exists(
            embedding_save_file_dir), f"embedding save file directory: '{embedding_save_file_dir}' is wrong."

        # self.embedding_network.load_state_dict(torch.load(embedding_save_file_dir))
        
        embedding_network_checkpoint = torch.load(embedding_save_file_dir)
        self.embedding_network.m_embedding.weight.data = embedding_network_checkpoint['m_embedding.weight']
        self.embedding_network.u_embedding.weight.data = embedding_network_checkpoint['u_embedding.weight']

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave.eval()
        
        self.srm_ave([torch.zeros((1, 100)), torch.zeros((1, state_size, 100))]) #[a,s]

        # PER
        self.buffer = PriorityExperienceReplay(
            self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1) / 500000
        self.std = 1.5

        self.is_test = is_test

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

        items_ebs = self.embedding_network.m_embedding(torch.tensor(items_ids, dtype=torch.long))#.detach().numpy()
        action = torch.transpose(action, 0, 1)
        
        if top_k:
            item_indice = np.argsort(torch.matmul(items_ebs, action).squeeze().detach().numpy())[-top_k:]
            return items_ids[item_indice]
        
        else:
            item_idx = torch.argmax(torch.matmul(items_ebs, action)).item()
            return items_ids[item_idx]

    def train(self, max_episode_num, top_k=False, load_model=False):
        # Initialize target networks
        self.actor.build_networks()
        self.critic.build_networks()
        
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            self.load_model(ROOT_DIR + "/save_weights/actor_50000.pth",
                            ROOT_DIR + "/save_weights/critic_50000.pth")
            print('Completely load weights!')
            time.sleep(3)

        episodic_precision_history = []

        for episode in range(max_episode_num):
            
            # for name, param in self.srm_ave.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name}: {param.data}")
        
            # Reset episodic reward
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0
            
            # Reset environment
            user_id, items_ids, done = self.env.reset()

            while not done:

                user_eb = self.embedding_network.u_embedding(torch.tensor([user_id], dtype=torch.long)).detach().numpy()
                items_eb = self.embedding_network.m_embedding(torch.tensor(items_ids, dtype=torch.long)).detach().numpy()

                state = self.srm_ave([
                    torch.tensor(user_eb, dtype=torch.float32), 
                    torch.tensor(items_eb, dtype=torch.float32).unsqueeze(0)
                ])
                
                # Get action (ranking score)
                action = self.actor.network(state)

                # ε-greedy exploration
                if self.epsilon > np.random.uniform() and not self.is_test:
                    self.epsilon -= self.epsilon_decay
                    action += torch.tensor(np.random.normal(0, self.std, size=action.shape), dtype=torch.float32)

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
                next_items_eb = self.embedding_network.m_embedding(torch.tensor(next_items_ids, dtype=torch.long)).detach().numpy()
                next_state = self.srm_ave([
                    torch.tensor(user_eb, dtype=torch.float32), 
                    torch.tensor(next_items_eb, dtype=torch.float32).unsqueeze(0)
                ])

                # Store in buffer
                action = action.detach().numpy()
                state = state.detach().numpy()
                next_state = next_state.detach().numpy()
                
                self.buffer.append(state, action, reward, next_state, done)

                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    # Sample a minibatch
                    batch = self.buffer.sample(self.batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = batch

                    # Set TD targets
                    target_next_action = self.actor.target_network(batch_next_states)
                    qs = self.critic.network([target_next_action, batch_next_states])
                    target_qs = self.critic.target_network([target_next_action, batch_next_states])
                    min_qs = torch.min(torch.cat([target_qs, qs], dim=1), dim=1, keepdim=True).values  # Double Q method
                    td_targets = self.calculate_td_target(batch_rewards, min_qs.detach().numpy(), batch_dones)

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
        