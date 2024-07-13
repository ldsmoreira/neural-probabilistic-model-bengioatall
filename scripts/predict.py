import torch
from common import init_model

def predict(model, inputs, device="cpu"):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs