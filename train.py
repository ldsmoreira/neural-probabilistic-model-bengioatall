import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import SentenceDataset
from model import NeuralProbabilisticModel

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)

    torch_inputs = []
    torch_labels = []

    for input, label in zip(inputs, labels):
        temp_input = torch.tensor([sentence_dataset.vocab[token] for token in input]).long()
        temp_label = torch.tensor(sentence_dataset.vocab[label]).long()

        torch_inputs.append(temp_input)
        torch_labels.append(temp_label)

    return torch.stack(torch_inputs), torch.stack(torch_labels)


# Initialize the dataset and dataloader
sentence_dataset = SentenceDataset("data/raw/ceten.xml", n_gram=3, vocab_size=50000)

total_size = len(sentence_dataset)
train_size = int(total_size * 0.8)
test_size = total_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(sentence_dataset, [train_size, test_size])

# Create DataLoaders for training and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Initialize the model
model = NeuralProbabilisticModel(sentence_dataset.vocab, embedding_dim=128, hidden_dim=256, n_gram=3)

def train(model, train_loader, optimizer, loss_function, num_epochs, device):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        total_loss = 0
        
        for inputs, labels in train_loader:
            print("Iteration")
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

def evaluate(model, test_loader, loss_function, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader)}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    
    train(model, train_loader, optimizer, loss_function, num_epochs=10, device=device)
    evaluate(model, test_loader, loss_function, device)