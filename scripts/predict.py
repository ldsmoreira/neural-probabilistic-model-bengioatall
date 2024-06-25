import torch
from common import init_model

def predict(model, inputs, device="cpu"):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

# Example usage
if __name__ == "__main__":
    # Assuming inputs are prepared and model is loaded
    pass