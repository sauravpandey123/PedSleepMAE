import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F


def train(model, device, train_loader, optimizer):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_tensor = batch["x"].squeeze().float().to(device) 
        optimizer.zero_grad()  # Zero gradients
        output, mask = model(input_tensor)
        loss = F.mse_loss(output, input_tensor)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, device, val_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(val_loader):
            input_tensor = batch["x"].squeeze().float().to(device) 
            output, mask = model(input_tensor)
            loss = F.mse_loss(output, input_tensor)
            running_loss += loss.item()
            
    avg_loss = running_loss / len(val_loader)
    return avg_loss


def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    f_path = os.path.join(checkpoint_dir, f"{filename}.pt")
    torch.save(state, f_path)
