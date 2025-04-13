import os
import random
import string
from typing import Counter
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from base import BaseDataLoader
from utils.text_processor import TextProcessor
import torch


class NGramDataset(Dataset):
    def __init__(self, data_dir, n=3, load_fraction=1.0, random_load=True, vocab_size=10000):
        """
        Dataset for generating n-grams from text content in XML files.
        Args:
            data_dir (str): Directory containing XML files.
            n (int): N-gram size.
            load_fraction (float): Fraction of files to load (0.0 - 1.0).
            random_load (bool): Whether to load files in random order.
        """
        if not (0 <= load_fraction <= 1):
            raise ValueError("load_fraction must be between 0 and 1.")
        
        self.processor = TextProcessor()

        self.ngrams = []

        self.vocab = None

        self.vocab_size = vocab_size

        self.word_to_index = None

        self.load_data(data_dir, n, load_fraction, random_load)

        self.build_vocab()

    def load_data(self, data_dir, n, load_fraction, random_load):
        """
        Load XML files, parse content, and generate n-grams.
        Args:
            data_dir (str): Directory containing XML files.
            n (int): N-gram size.
            load_fraction (float): Fraction of files to load (0.0 - 1.0).
            random_load (bool): Whether to load files in random order.
        """
        file_paths = []
        for root_dir, _, files in os.walk(data_dir):
            for file_name in files:
                file_paths.append(os.path.join(root_dir, file_name))

        # Shuffle file paths if random_load is True
        if random_load:
            random.shuffle(file_paths)

        # Select a subset of files based on load_fraction
        num_files_to_load = max(1, int(len(file_paths) * load_fraction))
        file_paths = file_paths[:num_files_to_load]

        for file_path in file_paths:
            # Process only XML files
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Wrap the content in a root tag if not well-formed XML
                content = f"<root>{content}</root>"

                # Parse the content
                root = ET.fromstring(content)

                # Extract documents and generate n-grams
                for doc in root.findall("doc"):
                    text_content = doc.text.strip() if doc.text else ""
                    cleaned_text = self.processor.preprocess(text_content)
                    self.ngrams += self.generate_ngrams_with_masks(cleaned_text, n)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    def build_vocab(self):
        """
        Build a vocabulary with tokens as keys and their indices as values.
        The last index is reserved for the UNK token.
        """
        # Count word frequencies
        word_counter = Counter()
        for ngram, target in self.ngrams:
            word_counter.update(ngram.split())  # Count words in n-grams
            word_counter[target] += 1          # Count the target word

        # Get the most common words up to the vocab_size minus one
        most_common_words = word_counter.most_common(self.vocab_size - 1)
        vocab = [word for word, _ in most_common_words]

        # Map words to indices, starting from 0
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}

        # Add the UNK token with the last index
        self.word_to_index["<UNK>"] = len(self.word_to_index)

        # Save the vocabulary list (optional, for reference)
        self.vocab = vocab

    def get_word_index(self, word):
        """
        Get the index of a word in the vocabulary.
        Returns the index of the special token for out-of-vocabulary words if the word is not found.
        """
        return self.word_to_index.get(word, self.word_to_index["<UNK>"])
    

    def get_vocab(self):
        return self.vocab
    
    def get_word_to_index(self):
        return self.word_to_index


    @staticmethod
    def generate_ngrams_with_masks(content, n):
        """Generate n-grams and targets with masks."""
        words = ["[BEG]"] * (n - 1) + content.split() + ["[END]"]
        ngrams = []
        for i in range(len(words) - n):
            ngram = " ".join(words[i:i+n])
            target = words[i+n]
            ngrams.append((ngram, target))
        return ngrams

    def __len__(self):
        return len(self.ngrams)

    # Modify __getitem__ temporarily for debugging
    def __getitem__(self, idx):
        ngram, target = self.ngrams[idx]
        return ngram, target  # Ensure only two items are returned


class NGramDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, n=3, load_fraction=1.0, random_load=True, vocab_size=10000):
        """
        DataLoader for the NGramDataset.
        Args:
            data_dir (str): Directory containing XML files.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            validation_split (float): Fraction of the data to use for validation.
            num_workers (int): Number of subprocesses for data loading.
            n (int): N-gram size.
            load_fraction (float): Fraction of files to load (0.0 - 1.0).
            random_load (bool): Whether to load files in random order.
        """
        self.dataset = NGramDataset(data_dir, n=n, load_fraction=load_fraction, random_load=random_load, vocab_size=vocab_size)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def collate_fn(batch, word_to_index, embedding_layer=None):
    """
    Collate function for converting words to indices or embeddings.

    Args:
        batch: List of tuples (ngram, target), where ngram is a string and target is the label.
        word_to_index: Dictionary mapping words to indices.
        embedding_layer: (Optional) PyTorch Embedding layer to convert indices to embeddings.

    Returns:
        context_tensor: Tensor of context word indices or embeddings.
        target_tensor: Tensor of target word indices.
    """
    
    contexts = []
    targets = []
    for ngram, target in zip(*batch):
        words = ngram.split()
        context_words = words

        context_indices = [word_to_index.get(w, word_to_index["<UNK>"]) for w in context_words]
        target_index = word_to_index.get(target, word_to_index["<UNK>"])  

        contexts.append(context_indices)
        targets.append(target_index)

    context_tensor = torch.tensor(contexts, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    
    return context_tensor, target_tensor




# Usage Example
if __name__ == "__main__":
    # Specify the data directory containing XML files
    data_dir = "../data/raw/ptwiki-latest-pages-articles"

    # Initialize DataLoader
    dataloader = NGramDataLoader(
        data_dir=data_dir,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Adjust based on system capability
        n=3,
        load_fraction=0.5,  # Load 50% of the files
        random_load=True  # Load files randomly
    )

    # Iterate through the DataLoader
    for ngrams, targets in dataloader:
        print("NGrams:", ngrams)
        print("Targets:", targets)
