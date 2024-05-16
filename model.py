import torch
import torch.nn as nn

# Parameters
vocab_size = 10  # Number of unique tokens in the vocabulary
embedding_dim = 5  # Dimension of the embedding vector

# Create an embedding layer
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
breakpoint()