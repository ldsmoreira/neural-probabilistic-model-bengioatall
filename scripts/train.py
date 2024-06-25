import torch
from torch import optim
import torch.nn.functional as F
from common import load_data, init_model, custom_collate_fn

def train(model, train_loader, optimizer, loss_function, num_epochs, device):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, labels)  # Calculate loss
            total_loss += loss.item()
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            if (batch_idx + 1) % 100 == 0:  # Log every 'log_interval' batches
                print(f"Epoch {epoch+1} [{batch_idx + 1}/{len(train_loader)}]: Batch Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = init_model(train_loader.dataset.dataset.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    train(model, train_loader, optimizer, loss_function, num_epochs=10, device=device)