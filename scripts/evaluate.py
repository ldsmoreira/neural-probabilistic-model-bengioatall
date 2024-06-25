import torch
from common import load_data, init_model

def evaluate(model, test_loader, loss_function, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = init_model(train_loader.dataset.dataset.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    evaluate(model, test_loader, loss_function, device)
