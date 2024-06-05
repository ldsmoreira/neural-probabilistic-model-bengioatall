from collections import Counter
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
    
class Vocabulary:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vocabulary = self._build_vocabulary(self.sentences)

    @staticmethod
    def tokenize(text):
        return text.lower().split()

    # Despite not being explicit used in the paper
    # It is a common practice to use a vocabulary sorted by frequency
    def _build_vocabulary(self, sentences):
        counter = Counter()
        for text in sentences:
            counter.update(self.tokenize(text))
        # Sort tokens by frequency in descending order and return the 50k most common tokens
        sorted_tokens_by_freq = [token for token in counter.most_common(50000)]
        return sorted_tokens_by_freq
    
    def __len__(self):
        return len(self.vocabulary)
    
    def __getitem__(self, idx):
        return self.vocabulary[idx]



if __name__ == "__main__":
    sentence_dataset = SentenceDataset("data/raw/ceten.xml")
    vocab = Vocabulary(sentence_dataset)
    # 20 most common portuguese tokens
    print(vocab[:20])