import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SentenceDataset
from model import NeuralProbabilisticModel

def custom_collate_fn(batch):

    inputs, labels = zip(*batch)

    torch_inputs = []
    torch_labels = []

    for input, label in zip(inputs, labels):

        temp_input = torch.Tensor([sentence_dataset.vocab[token] for token in input]).long()
        temp_label = torch.Tensor([sentence_dataset.vocab[label]]).long()

        torch_inputs.append(temp_input)
        torch_labels.append(temp_label)
        

    return torch.stack(torch_inputs), F.one_hot(torch.stack(torch_labels), num_classes=len(sentence_dataset.vocab))

sentence_dataset = SentenceDataset("data/raw/ceten.xml")
dataloader = DataLoader(sentence_dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
model = NeuralProbabilisticModel(sentence_dataset.vocab, embedding_dim=128, hidden_dim=256)

def train():
    for i, (inputs, labels) in enumerate(dataloader):
        model(inputs)

if __name__ == "__main__":
    train()