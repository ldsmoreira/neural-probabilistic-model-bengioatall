import torch
from torch.utils.data import DataLoader, random_split
from data.dataset import SentenceDataset
from models.model import NeuralProbabilisticModel

sentence_dataset = SentenceDataset("data/raw/ceten.xml", n_gram=3, vocab_size=50000)

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    torch_inputs = [torch.tensor([sentence_dataset.vocab[token] for token in input]).long() for input in inputs]
    torch_labels = [torch.tensor(sentence_dataset.vocab[label]).long() for label in labels]
    return torch.stack(torch_inputs), torch.stack(torch_labels)

def load_data():
    total_size = len(sentence_dataset)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(sentence_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, test_loader

def init_model(vocab):
    return NeuralProbabilisticModel(vocab, embedding_dim=100, hidden_dim=32, n_gram=3)
