from dataset import SentenceDataset, Vocabulary
from model import NeuralProbabilisticModel

sentence_dataset = SentenceDataset("data/raw/ceten.xml")
vocab = Vocabulary(sentence_dataset.sentences)

model = NeuralProbabilisticModel(vocab, 128, 256)
breakpoint()