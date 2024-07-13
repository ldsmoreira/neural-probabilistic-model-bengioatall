import re
from collections import Counter
from torch.utils.data import Dataset

# Helper function to split words and punctuation
def tokenize(text):
    return re.findall(r'\b\w+\b|[^\w\s]', text.lower())

# Sentence Dataset
class SentenceDataset(Dataset):
    def __init__(self, path, n_gram=3):
        self.sentences = self._load_sentences(path)
        self.vocab = Vocabulary(self.sentences)
        self.n_gram = n_gram
        self.data, self.label = self._build_dataset(self.n_gram)

    def _load_sentences(self, path):
        with open(path, "r") as file:
            text = file.read()
        sentences = text.replace("<s> ", "").replace(" </s> ", "").split("\n")
        return sentences
    
    def _build_dataset(self, n_gram):
        data = []
        label = []
        for sentence in self.sentences:
            tokens = tokenize(sentence)
            if len(tokens) >= n_gram + 1:
                for i in range(len(tokens) - n_gram):
                    data.append([self.vocab[token] for token in tokens[i:i + n_gram]])
                    label.append(self.vocab[tokens[i + n_gram]])
        return data, label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
class Vocabulary:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vocabulary = self._build_vocabulary()

    def _build_vocabulary(self):
        counter = Counter()
        for sentence in self.sentences:
            counter.update(tokenize(sentence))
        # Assign index to frequent tokens and reserve the last index for <UNK>
        vocab_size = len({token for token, freq in counter.items() if freq >= 40})
        sorted_tokens_by_freq = {token[0]: index for index, token in enumerate(counter.most_common(vocab_size))}
        sorted_tokens_by_freq["<UNK>"] = vocab_size  # Ensure <UNK> has a unique index
        return sorted_tokens_by_freq
    
    def __len__(self):
        return len(self.vocabulary)
    
    def __getitem__(self, token):
        if type(token) == int:
            return list(self.vocabulary.keys())[token]
        return self.vocabulary.get(token, self.vocabulary["<UNK>"])

if __name__ == "__main__":
    sentence_dataset = SentenceDataset("data/raw/ceten.xml", n_gram=3)
    vocab = sentence_dataset.vocab
    print(list(vocab.vocabulary.items())[:20])  # Print 20 most common tokens
    breakpoint()
