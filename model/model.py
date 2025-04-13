# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralProbabilisticLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        """
        Initialize the Neural Probabilistic Language Model.
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            context_size (int): Number of context words (n-1).
            hidden_dim (int): Number of hidden units in the hidden layer.
        """
        super(NeuralProbabilisticLanguageModel, self).__init__()
        
        # Embedding layer to learn word representations
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layer (input size is embedding_dim * context_size)
        self.hidden = nn.Linear(embedding_dim * context_size, hidden_dim)
        
        # Output layer (hidden_dim to vocab_size)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, context):
        """
        Forward pass through the model.
        Args:
            context (Tensor): Tensor of shape (batch_size, context_size) containing word indices.
        Returns:
            logits (Tensor): Tensor of shape (batch_size, vocab_size) containing raw scores for each word.
        """
        # Lookup embeddings for context words
        embeddings = self.embeddings(context)  # (batch_size, context_size, embedding_dim)
        
        # Flatten embeddings to feed into the hidden layer
        flattened = embeddings.view(embeddings.size(0), -1)  # (batch_size, context_size * embedding_dim)
        
        # Pass through hidden layer with tanh activation
        hidden_out = torch.tanh(self.hidden(flattened))  # (batch_size, hidden_dim)
        
        # Apply dropout
        hidden_out = self.dropout(hidden_out)
        
        # Pass through output layer
        logits = self.output(hidden_out)  # (batch_size, vocab_size)
        
        return logits
