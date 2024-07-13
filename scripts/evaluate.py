import torch
from common import load_data, init_model
from models.model import NeuralProbabilisticModel

def calculate_similarity(model : NeuralProbabilisticModel, word1, word2, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        word1_index = torch.tensor([model.vocab[word1]]).to(device)
        word2_index = torch.tensor([model.vocab[word2]]).to(device)
        word1_embedding = model.embeddings(word1_index)
        word2_embedding = model.embeddings(word2_index)
        similarity = torch.nn.functional.cosine_similarity(word1_embedding, word2_embedding)
    print(f"Similarity between '{word1}' and '{word2}': {similarity.item()}")

def evaluate(model, test_loader, loss_function, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
        # Partial evaluation

    print(f"Test Loss: {total_loss / len(test_loader)}")

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = init_model(test_loader.dataset.dataset.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    evaluate(model, test_loader, loss_function, device)
    calculate_similarity(model, "carro", "pneu", device)
