import torch
import torch.nn as nn

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


if __name__ == "__main__":
    movie = MovieGenreEmbedding(10, 5, 3)