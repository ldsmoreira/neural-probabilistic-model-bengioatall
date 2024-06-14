from torch.utils.data import DataLoader
from dataset import SentenceDataset, Vocabulary
from model import NeuralProbabilisticModel

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    return inputs, labels

def train():
    sentence_dataset = SentenceDataset("data/raw/ceten.xml")
    dataloader = DataLoader(sentence_dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    model = NeuralProbabilisticModel(sentence_dataset.vocab, embedding_dim=128, hidden_dim=256)

    for i, (inputs, labels) in enumerate(dataloader):
        breakpoint()

if __name__ == "__main__":
    train()