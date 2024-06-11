import torch
import torch.nn as nn


# MINWOO TODO 
# self.N : item 개수 argument 를 추가해야함.
# average pooling 하는 순간, vector 를 scalar 로 바꾸기 때문에, 정보 손실이 매우 클 것 같은데?


class DRRAveStateRepresentation(nn.Module):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim        
        self.wav = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        
        self.weights = torch.tensor([0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01])
        
    def forward(self, x):
        
        # items_eb = x[1] / self.embedding_dim 
        # avp = self.wav(items_eb).squeeze(1)
        items_eb = x[1] 
        
        for i in range(10):
            items_eb[:,i,:] = items_eb[:,i,:] * self.weights[i]
        
        avp = torch.sum(items_eb, dim=1).squeeze(1)
        user_avp = x[0] * avp
        concat = torch.cat([x[0], user_avp, avp],dim=1)
        return concat



if __name__ == '__main__':
    embedding_dim = 128
    model = DRRAveStateRepresentation(embedding_dim)
    x0 = torch.rand(32, embedding_dim)  # Example user embeddings
    x1 = torch.rand(32, 10, embedding_dim)  # Example item embeddings
    output = model([x0, x1])
    print(output.shape)
    for name, param in model.named_parameters():
        print(name)
    