import re
import torch
from torch.utils.data import Dataset, DataLoader

# Sentence Dataset
class SentenceDataset(Dataset):
    def __init__(self, path):
        self.sentences = self._load_sentences(path)

    def _load_sentences(self, path):
        with open(path, "r") as file:
            text = file.read()

        sentences = text.replace("<s> ", "").replace(" </s> ", "").split("\n")

        return sentences

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]


if __name__ == "__main__":
    sentence_dataset = SentenceDataset("data/raw/ceten.xml")
    breakpoint()