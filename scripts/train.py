import os
import torch
from torch import optim
import torch.nn.functional as F
from common import load_data, init_model, custom_collate_fn
from scripts import evaluate

def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {checkpoint_path}, Epoch: {epoch}")
        return epoch
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

def train(model, train_loader, test_loader, optimizer, loss_function, num_epochs, device, checkpoint_path=None):
    start_epoch = 0
    if checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path) + 1
    
    with open("losses.txt", "w") as f:
        model.train()  # Set the model to training mode
        for epoch in range(start_epoch, num_epochs):
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

                if (batch_idx + 1) % 1000 == 0:  # Log every 'log_interval' batches
                    print(f"Epoch {epoch+1} [{batch_idx + 1}/{len(train_loader)}]: Batch Loss: {loss.item():.4f}")
                    f.write(f"Epoch {epoch+1} [{batch_idx + 1}/{len(train_loader)}]: Batch Loss: {loss.item():.4f}\n")

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

            evaluate.evaluate(model, test_loader, loss_function, device)
            evaluate.calculate_similarity(model, "carro", "pneu", device)
            evaluate.calculate_similarity(model, "pneu", "gato", device)
            evaluate.calculate_similarity(model, "gato", "cachorro", device)
            evaluate.calculate_similarity(model, "cachorro", "carro", device)
            
            save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = init_model(train_loader.dataset.dataset.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    # Specify the checkpoint file you want to load, or set to None to start from scratch
    checkpoint_path = "checkpoints/model_epoch_10.pth"  # Change this path as needed, or set to None to start from scratch

    train(model, train_loader, test_loader, optimizer, loss_function, num_epochs=10, device=device, checkpoint_path=checkpoint_path)
