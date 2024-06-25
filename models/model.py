import torch
import torch.nn as nn

from data.dataset import Vocabulary

class NeuralProbabilisticModel(nn.Module):
    def __init__(self, vocab: Vocabulary, embedding_dim, hidden_dim, n_gram=3):
        super(NeuralProbabilisticModel, self).__init__()
        self.vocab = vocab

        # Embedding layer
        self.C = nn.Embedding(len(vocab), embedding_dim)

        # Hidden layer
        self.dense1 = nn.Linear(embedding_dim * n_gram, hidden_dim)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.constant_(self.dense1.bias, 0)

        # Output layer
        self.dense2 = nn.Linear(hidden_dim, len(vocab))
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.constant_(self.dense2.bias, 0)
    
    def forward(self, x):

        out = self.C(x)
        # Stack n-gram embeddings
        out = out.view(out.size(0), -1)
        out = torch.tanh(self.dense1(out))
        out = self.dense2(out)
        return out
    
    def get_token_embedding(self, token):
        idx = self.vocab[token]
        return self.C(torch.tensor(idx))