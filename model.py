import torch
import torch.nn as nn

from dataset import Vocabulary

class NeuralProbabilisticModel(nn.Module):
    def __init__(self, vocab: Vocabulary, embedding_dim, hidden_dim):
        super(NeuralProbabilisticModel, self).__init__()
        self.vocab = vocab

        # Embedding layer
        self.C = nn.Embedding(len(vocab), embedding_dim)
        self.dense1 = nn.Linear(embedding_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, len(vocab))
    
    def forward(self, x):
        # x is a tensor of n * num_features x 1
        out = self.C(x)
        out = torch.tanh(self.dense1(out))
        out = torch.softmax(self.dense2(x))
        return x
    
    def get_token_embedding(self, token):
        idx = self.vocab[token]
        return self.C(torch.tensor(idx))