import torch
import torch.nn as nn

class NeuralProbabilisticModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(NeuralProbabilisticModel, self).__init__()
        # Embedding layer
        self.C = nn.Embedding(vocab_size, embedding_dim)
        self.dense1 = nn.Linear(embedding_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        out = self.C(x)
        out = torch.tanh(self.dense1(out))
        out = torch.softmax(self.dense2(x))
        return x