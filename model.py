import torch
import torch.nn as nn

from dataset import Vocabulary

class NeuralProbabilisticModel(nn.Module):
    def __init__(self, vocab: Vocabulary, embedding_dim, hidden_dim):
        super(NeuralProbabilisticModel, self).__init__()
        self.vocab = vocab

        # Embedding layer
        self.C = nn.Embedding(len(vocab), embedding_dim)

        # Hidden layer
        self.dense1 = nn.Linear(embedding_dim, hidden_dim)

        # Output layer
        self.dense2 = nn.Linear(hidden_dim, len(vocab))
    
    def forward(self, x):

        out = self.C(x)
        breakpoint()
        out = torch.tanh(self.dense1(out))
        out = torch.softmax(self.dense2(out))
        return x
    
    def get_token_embedding(self, token):
        idx = self.vocab[token]
        return self.C(torch.tensor(idx))