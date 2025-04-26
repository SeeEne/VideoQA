# Filename: train.py
# Purpose: Contains helper functions for training and evaluating probe models
#          on PRE-COMPUTED embeddings.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm # Or from tqdm import tqdm

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader, # Expects (embedding, label) batches
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch_num: int = 0,
                    total_epochs: int = 0) -> tuple[float, float]:
    """
    Performs one full training pass over the pre-computed embedding dataset.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Train]", leave=False)

    for embeddings, labels in progress_bar: # Iterate through embedding batches
        embeddings = embeddings.to(device, dtype=torch.float32) # Embeddings to device
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(embeddings) # Pass embeddings to probe
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * embeddings.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0: return 0.0, 0.0
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100.0
    print(f"Epoch {epoch_num+1}/{total_epochs} [Train] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module,
            dataloader: DataLoader, # Expects (embedding, label) batches
            criterion: nn.Module,
            device: torch.device) -> tuple[float, float]:
    """
    Evaluates the probe model on a pre-computed embedding dataset.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for embeddings, labels in progress_bar: # Iterate through embedding batches
            embeddings = embeddings.to(device, dtype=torch.float32) # Embeddings to device
            labels = labels.to(device, dtype=torch.long)

            outputs = model(embeddings) # Pass embeddings to probe
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0: return 0.0, 0.0
    avg_loss = running_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100.0
    print(f"Evaluation Results: Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")
    return avg_loss, accuracy