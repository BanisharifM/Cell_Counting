import optuna
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from glob import glob
from tqdm import tqdm
from dataset_handler import CellDataset
from model import CellCounter
from denseweight import DenseWeight
import numpy as np
import matplotlib.pyplot as plt

# Set the output directory for saving models and plots
# output_dir = "Experiments/39/"
# os.makedirs(output_dir, exist_ok=True)

def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    cell_locations = [loc.clone().detach() for loc in cell_locations]
    padded_cell_locations = pad_sequence(cell_locations, batch_first=True, padding_value=-1)
    return images, labels, padded_cell_locations

def get_data_loaders(batch_size=16):
    train_image_paths = glob("IDCIA_Augmentated_V2/images/train/*.tiff")
    val_image_paths = glob("IDCIA_Augmentated_V2/images/val/*.tiff")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    return train_loader, val_loader

def calculate_metrics(pred_count, true_count):
    """Calculate corrected metrics."""
    mae = torch.abs(pred_count - true_count).mean().item()
    rmse = torch.sqrt(torch.mean((pred_count - true_count) ** 2)).item()
    percentage_accuracy = (1 - (torch.abs(pred_count - true_count) / true_count)).clamp(0, 1).mean().item() * 100
    return mae, rmse, percentage_accuracy

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=5e-4, alpha=1.0, weight_decay=1e-5, patience=50):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    dw = DenseWeight(alpha=alpha)

    best_val_loss = float('inf')
    best_epoch = -1
    train_losses, val_losses = [], []
    train_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}
    val_metrics = {"MAE": [], "RMSE": [], "PercentageAccuracy": []}
    early_stop_counter = 0
    total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, cell_locations in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count, predicted_locations = model(inputs)

            # Calculate location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):
                true_loc = cell_locations[j].to(device)
                num_locations = true_loc.size(0)
                pred_loc = predicted_locations[j].view(-1, 2)[:num_locations]
                dists = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists, min_indices = dists.min(dim=1)
                location_loss = F.mse_loss(pred_loc, true_loc[min_indices])
                total_location_loss += location_loss

            # Count loss and total loss
            count_loss = criterion(cell_count, labels).mean()
            total_loss = count_loss + total_location_loss / inputs.size(0)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            
            # Metrics calculations
            mae, rmse, percentage_accuracy = calculate_metrics(cell_count, labels)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_percentage_accuracy += percentage_accuracy * inputs.size(0)

        # Log epoch loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics["MAE"].append(total_mae / len(train_loader.dataset))
        train_metrics["RMSE"].append(total_rmse / len(train_loader.dataset))
        train_metrics["PercentageAccuracy"].append(total_percentage_accuracy / len(train_loader.dataset))

        # Evaluate on validation set
        val_loss, val_mae, val_rmse, val_percentage_accuracy = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_metrics["MAE"].append(val_mae)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["PercentageAccuracy"].append(val_percentage_accuracy)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1  # Save the epoch (1-based indexing)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Best model updated at epoch {best_epoch} with Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

    return model, train_losses, val_losses, train_metrics, val_metrics, best_epoch, best_val_loss


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_mae, total_rmse, total_percentage_accuracy = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, labels, cell_locations in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count, predicted_locations = model(inputs)

            # Count loss
            count_loss = criterion(cell_count, labels).mean()

            # Location loss
            total_location_loss = 0
            for j in range(inputs.size(0)):
                true_loc = cell_locations[j].to(device)
                num_locations = true_loc.size(0)
                pred_loc = predicted_locations[j].view(-1, 2)[:num_locations]
                dists = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists, min_indices = dists.min(dim=1)
                location_loss = F.mse_loss(pred_loc, true_loc[min_indices])
                total_location_loss += location_loss

            total_loss += count_loss + total_location_loss / inputs.size(0)

            # Metrics calculations
            mae, rmse, percentage_accuracy = calculate_metrics(cell_count, labels)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_percentage_accuracy += percentage_accuracy * inputs.size(0)

    return total_loss / len(data_loader), total_mae / len(data_loader.dataset), total_rmse / len(data_loader.dataset), total_percentage_accuracy / len(data_loader.dataset)

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    patience = trial.suggest_int("patience", 10, 50)
    num_epochs = trial.suggest_int("num_epochs", 50, 300)

    train_loader, val_loader = get_data_loaders(batch_size)
    model = CellCounter()

    _, _, _, _, val_metrics, _, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience
    )

    return best_val_loss

def run_optuna():
    study = optuna.create_study(direction="minimize", storage=None)  # Use in-memory storage
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

    # Save best results
    with open("best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f)
    print("Best parameters and results saved to best_params.json")

if __name__ == "__main__":
    run_optuna()
