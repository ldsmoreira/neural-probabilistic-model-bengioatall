from collections import Counter
from torch.utils.data import Dataset

# Sentence Dataset
class SentenceDataset(Dataset):
    def __init__(self, path, n_gram=3):
        self.sentences = self._load_sentences(path)
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
        sorted_tokens_by_freq = {token[0]: index for index, token in enumerate(counter.most_common(50000))}
        return sorted_tokens_by_freq
    
    def __len__(self):
        return len(self.vocabulary)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return list(self.vocabulary.keys())[idx]
        elif isinstance(idx, str):
            return self.vocabulary[idx]



if __name__ == "__main__":
    sentence_dataset = SentenceDataset("data/raw/ceten.xml")
    vocab = Vocabulary(sentence_dataset)
    # 20 most common portuguese tokens
    print(vocab[:20])