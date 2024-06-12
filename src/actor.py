import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        '''
        Actor network consists of 3 fully connected layers.
        
            3k -> FC[ReLU] -> h
            h  -> FC[ReLU] -> h
            h  -> FC[Tanh] -> k
        
        '''
        
        self.fc = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )    
    def forward(self, x):
        return self.fc(x)
    
    
class Actor:
    
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 learning_rate, 
                 tau):
                
        self.embedding_dim = embedding_dim
        
        # Actor network and target network
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Soft target network update hyperparameter
        self.tau = tau
    
    def build_networks(self):
        # Build networks (dummy forward pass to initialize)
        self.network(torch.zeros(1, 3 * self.embedding_dim)) # input tensor.shape = (1, 3k)
        self.target_network(torch.zeros(1, 3 * self.embedding_dim))
    
    def update_target_network(self):
        # Soft target network update
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def train(self, states, dq_das):
        self.optimizer.zero_grad()
        outputs = self.network(states) # outputs.shape = (batch_size, k) -> a
        loss = -(outputs * dq_das).mean() #dq_das? 
        loss.backward() # (outputs * dq_das).shape = (batch_size, k)
        self.optimizer.step()
        
    def save_weights(self, path: str):
        torch.save(self.network.state_dict(), path)
        
    def load_weights(self, path: str):
        self.network.load_state_dict(torch.load(path))
        

if __name__ == '__main__':
    
    embedding_dim = 128
    hidden_dim = 256
    
    learning_rate = 1e-3
    state_size = 3 * embedding_dim
    tau = 1e-3
    
    actor = Actor(embedding_dim, hidden_dim, learning_rate, state_size, tau)
    actor.build_networks()
    
    x = torch.rand(32, 3 * embedding_dim)
    dq_das = torch.rand(32, embedding_dim)
    
    actor.train(x, dq_das)
    
    
