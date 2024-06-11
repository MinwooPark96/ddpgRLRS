import torch
import torch.nn as nn
import tensorflow as tf
import os 
import numpy as np

class MovieGenreEmbedding(nn.Module):
    def __init__(self, len_movies, len_genres, embedding_dim):
        super(MovieGenreEmbedding, self).__init__()
        
        # Embedding layers : trainable
        self.m_embedding = nn.Embedding(num_embeddings=len_movies, embedding_dim=embedding_dim)
        self.g_embedding = nn.Embedding(num_embeddings=len_genres, embedding_dim=embedding_dim)
        
        # Dot product layer
        self.m_g_merge = nn.CosineSimilarity(dim=1)
        
        # Output layer
        self.m_g_fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        memb = self.m_embedding(x[:, 0])
        gemb = self.g_embedding(x[:, 1])
        m_g = self.m_g_merge(memb, gemb).unsqueeze(1)
        
        return self.sigmoid(self.m_g_fc(m_g))

class UserMovieEmbedding(nn.Module):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        
        # Embedding layers : trainable
        self.u_embedding = nn.Embedding(num_embeddings=len_users, embedding_dim=embedding_dim)
        self.m_embedding = nn.Embedding(num_embeddings=len_movies, embedding_dim=embedding_dim)
        
        # Dot product layer
        self.m_u_merge = nn.CosineSimilarity(dim=1)
        
        # Output layer
        self.m_u_fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        user_ids, movie_ids = x[:, 0], x[:, 1]
        
        uemb = self.u_embedding(user_ids)
        memb = self.m_embedding(movie_ids)
        #uemb.shape = memb.shape = (batch_size, hidden_dim)
        m_u = (uemb * memb).sum(dim=1, keepdim=True) # It is Matrix Factorization Model
        
        #m_u.shape = batch,1
        return torch.sigmoid(self.m_u_fc(m_u))

class UserMovieMultiModalEmbedding(tf.keras.Model):
    def __init__(self, 
                 len_users, 
                 len_movies, 
                 embedding_dim, 
                 modality=('video', 'audio', 'text'), 
                 fusion='early', 
                 aggregation='concat'):
        
        super(UserMovieMultiModalEmbedding, self).__init__()
        self.modality = modality
        self.fusion = fusion
        self.aggregation = aggregation
        
        # input: (user, movie)
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        
        # user embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        
        # item embedding        
        if not modality:
            self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        
        else:
            # load multimodal features
            for mod in modality:
                ROOT_DIR = os.getcwd()
                DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m')
                mod_name = 'image' if mod == 'video' else mod # rename due to file name
                setattr(self, f'{mod}_feat', np.load(f'{DATA_DIR}/{mod_name}_feat.npy'))
                
            if fusion == 'early':
                self.mm_fc = tf.keras.layers.Dense(embedding_dim, name='mm_fc')
                
            elif fusion == 'late':
                if aggregation == 'concat':
                    def divide_integer(n, parts):
                        q, r = divmod(n, parts)
                        return [q+1]*(r) + [q]*(parts-r)
                    embedding_dims = divide_integer(embedding_dim, len(modality))
                elif aggregation == 'mean':
                    embedding_dims = [embedding_dim]*len(modality)
                    
                for i, mod in enumerate(modality):
                    setattr(self, f'{mod}_fc', tf.keras.layers.Dense(embedding_dims[i], name=f'{mod}_fc'))
        
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def get_embedding(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        
        if not self.modality:
            memb = self.m_embedding(x[1])
        else:
            mm_emb = []
            for mod in self.modality:
                mm_feat = getattr(self, f'{mod}_feat')
                x[1] = tf.cast(x[1], tf.int32)
                x[0] = tf.cast(x[0], tf.int32)
                mm_feat = tf.gather(mm_feat, x[1])
                
                if self.fusion == 'early':
                    mm_emb.append(mm_feat)
                elif self.fusion == 'late':
                    mm_emb.append(getattr(self, f'{mod}_fc')(mm_feat))
            
            if self.aggregation == 'concat':
                memb = tf.concat(mm_emb, axis=1)
            elif self.aggregation == 'mean':
                memb = tf.reduce_mean(tf.stack(mm_emb), axis=0)
                
            if self.fusion == 'early':
                memb = self.mm_fc(memb)
        return uemb, memb
        
    def call(self, x):
        uemb, memb = self.get_embedding(x)
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)

if __name__ == "__main__":
    pass