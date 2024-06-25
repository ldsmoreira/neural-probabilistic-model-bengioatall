from collections import Counter
from torch.utils.data import Dataset

# Sentence Dataset
class SentenceDataset(Dataset):
    def __init__(self, path, n_gram=3, vocab_size=50000):
        self.sentences = self._load_sentences(path)
        self.vocab = Vocabulary(self.sentences, vocab_size)
        self.n_gram = n_gram

        self.data, self.label = self._build_dataset(self.sentences, self.n_gram)

    def _load_sentences(self, path):
        with open(path, "r") as file:
            text = file.read()

        sentences = text.replace("<s> ", "").replace(" </s> ", "").split("\n")

        return sentences
    
    def _build_dataset(self, sentences, n_gram):
        data = []
        label = []
        for sentence in sentences:
            tokens = sentence.split()
            if len(tokens) >= n_gram + 1:
                for i in range(len(tokens) - n_gram):
                    data.append(tokens[i:i + n_gram])
                    label.append(tokens[i + n_gram])
        return data, label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
class Vocabulary:
    def __init__(self, sentences, vocab_size=50000):
        self.sentences = sentences
        self.vocab_size = vocab_size
        self.vocabulary = self._build_vocabulary(self.sentences)
        self.vocabulary["<UNK>"] = vocab_size  # Add <UNK> token at the end

    @staticmethod
    def tokenize(text):
        return text.lower().split()

    def _build_vocabulary(self, sentences):
        counter = Counter()
        for text in sentences:
            counter.update(self.tokenize(text))
        # Sort tokens by frequency in descending order and limit to vocab_size
        sorted_tokens_by_freq = {token[0]: index for index, token in enumerate(counter.most_common(self.vocab_size))}
        return sorted_tokens_by_freq
    
    def __len__(self):
        return len(self.vocabulary)
    
    def __getitem__(self, token):
        return self.vocabulary.get(token, self.vocabulary["<UNK>"])

if __name__ == "__main__":
    sentence_dataset = SentenceDataset("data/raw/ceten.xml", n_gram=3, vocab_size=50000)
    vocab = sentence_dataset.vocab
    # Print 20 most common tokens
    print(list(vocab.vocabulary.items())[:20])
