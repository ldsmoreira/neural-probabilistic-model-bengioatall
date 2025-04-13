import string
import unicodedata

class TextProcessor:
    def __init__(self):
        """
        Text preprocessor to clean and standardize text.
        Removes punctuation, accents, and converts to lowercase.
        """
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def remove_punctuation(self, text):
        """
        Remove punctuation from the text.
        Args:
            text (str): Input text.
        Returns:
            str: Text without punctuation.
        """
        return text.translate(self.punctuation_table)

    def remove_accents(self, text):
        """
        Remove accents from the text.
        Args:
            text (str): Input text.
        Returns:
            str: Text without accents.
        """
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )

    def to_lowercase(self, text):
        """
        Convert text to lowercase.
        Args:
            text (str): Input text.
        Returns:
            str: Lowercase text.
        """
        return text.lower()

    def preprocess(self, text):
        """
        Preprocess the text by removing punctuation, accents, and converting to lowercase.
        Args:
            text (str): Input text.
        Returns:
            str: Preprocessed text.
        """
        text = self.remove_punctuation(text)
        # text = self.remove_accents(text)
        text = self.to_lowercase(text)
        return text

# Usage Example
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_text = "Olá, mundo! Como você está? É incrível, não é?!"
    clean_text = preprocessor.preprocess(sample_text)
    print("Original:", sample_text)
    print("Preprocessed:", clean_text)
