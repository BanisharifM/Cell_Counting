'''
File: main.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that trains a cell counting model using the IDCIA dataset.

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_handler import CellDataset
from model import CellCounter
import numpy as np

def get_data_loaders(batch_size=8):
    '''
    Creates training and validation data loaders for the IDCIA dataset.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        train_loader (DataLoader): A DataLoader object for the training set.
        val_loader (DataLoader): A DataLoader object for the validation set.
    '''

    # Get training and testing image paths
    # train_image_paths = glob("IDCIA/images/*.tiff")
    # val_image_paths = glob("IDCIA/images/*.tiff")
    train_image_paths = glob("IDCIA_Augmentated/images/train/*.tiff")
    val_image_paths = glob("IDCIA_Augmentated/images/val/*.tiff")

    # Define transforms
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

    # Create datasets
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    '''
    Trains the cell counting model on the training set and evaluates it on the validation set.

    Args:
        model (CellCounter): The model to train.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        num_epochs (int): The number of epochs to train the model.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        model (CellCounter): The trained model.
        train_losses, val_losses (list): Lists of training and validation losses for each epoch.
        train_metrics, val_metrics (dict): Dict of lists of metrics (MAE, RMSE, ACP) for training and validation for each epoch.

    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss()  # This is MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For tracking losses and metrics
    train_losses, val_losses = [], []
    train_metrics = {"MAE": [], "RMSE": [], "ACP": []}
    val_metrics = {"MAE": [], "RMSE": [], "ACP": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_mae, total_rmse, total_acp = 0.0, 0.0, 0.0
        num_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics for this batch
            mae = torch.mean(torch.abs(outputs - labels)).item()  # Mean Absolute Error (MAE)
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()  # Root Mean Square Error (RMSE)

            # Acceptable Count Percent (ACP): Count of predictions within some threshold (e.g., 10% of true value)
            acceptable_error_threshold = 0.05 * torch.abs(labels)
            acp = torch.mean((torch.abs(outputs - labels) <= acceptable_error_threshold).float()).item()

            # Aggregate metrics
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_acp += acp * inputs.size(0)
            num_samples += inputs.size(0)

        # Store loss and metrics for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        train_metrics["MAE"].append(total_mae / num_samples)
        train_metrics["RMSE"].append(total_rmse / num_samples)
        train_metrics["ACP"].append(total_acp / num_samples)

        # Evaluate on validation set
        val_loss, val_mae, val_rmse, val_acp = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        val_metrics["MAE"].append(val_mae)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["ACP"].append(val_acp)

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAE: {train_metrics['MAE'][-1]:.4f}, Val MAE: {val_metrics['MAE'][-1]:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE'][-1]:.4f}, Val RMSE: {val_metrics['RMSE'][-1]:.4f}")
        print(f"Train ACP: {train_metrics['ACP'][-1]:.4f}, Val ACP: {val_metrics['ACP'][-1]:.4f}")

    return model, train_losses, val_losses, train_metrics, val_metrics

def evaluate_model(model, data_loader, criterion, device):
    '''
    Evaluates the model on a validation or test set.

    Args:
        model (CellCounter): The model to evaluate.
        data_loader (DataLoader): The DataLoader for the validation or test set.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to run the evaluation on.

    Returns:
        total_loss (float): The average loss over the dataset.
        mae (float): Mean Absolute Error (MAE) over the dataset.
        rmse (float): Root Mean Square Error (RMSE) over the dataset.
        acp (float): Acceptable Count Percent (ACP) over the dataset.
    '''
    model.eval()
    total_loss = 0.0
    total_mae, total_rmse, total_acp = 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate metrics
            mae = torch.mean(torch.abs(outputs - labels)).item()
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
            acceptable_error_threshold = 0.10 * torch.abs(labels)
            acp = torch.mean((torch.abs(outputs - labels) <= acceptable_error_threshold).float()).item()

            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_acp += acp * inputs.size(0)
            num_samples += inputs.size(0)

    return total_loss / len(data_loader), total_mae / num_samples, total_rmse / num_samples, total_acp / num_samples

def plot_losses(train_losses, val_losses):
    '''
    Plots the training and validation losses.

    Args:
        train_losses (list): A list of training losses for each epoch.
        val_losses (list): A list of validation losses for each epoch.
    '''

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def plot_metrics(train_metrics, val_metrics, metric_name):
    '''
    Plots the training and validation metrics over epochs.

    Args:
        train_metrics (list): A list of training metrics for each epoch.
        val_metrics (list): A list of validation metrics for each epoch.
        metric_name (str): The name of the metric being plotted.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.savefig(f'{metric_name.lower()}_plot.png')
    plt.close()

def main():
    # Set hyperparameters
    batch_size = 16
    num_epochs = 150
    learning_rate = 5e-5

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Create the model
    model = CellCounter()

    # Train the model
    trained_model, train_losses, val_losses, train_metrics, val_metrics = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

    # Plot each metric
    for metric in ["MAE", "RMSE", "ACP"]:
        plot_metrics(train_metrics[metric], val_metrics[metric], metric)

    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    
    test_loss, test_mae, test_rmse, test_acp = evaluate_model(trained_model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test ACP: {test_acp:.4f}")

    # Save the model
    torch.save(trained_model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()