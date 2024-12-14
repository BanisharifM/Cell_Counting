import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms import functional as TF
from glob import glob
from tqdm import tqdm
from dataset_handler import CellDataset
from model9 import CellCounter
from denseweight import DenseWeight
import numpy as np
import matplotlib.pyplot as plt

# Set output directory
output_dir = "Experiments/66"
os.makedirs(output_dir, exist_ok=True)

# Custom collate function
def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    padded_cell_locations = pad_sequence(cell_locations, batch_first=True, padding_value=-1)
    return images, labels, padded_cell_locations

# Custom data augmentations
class MicroscopyAugmentations:
    def __call__(self, img):
        if np.random.rand() > 0.5:
            img = TF.hflip(img)
        if np.random.rand() > 0.5:
            img = TF.vflip(img)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30)  # Random rotation
            img = TF.rotate(img, angle)
        if np.random.rand() > 0.5:
            img = TF.adjust_brightness(img, np.random.uniform(0.8, 1.2))
        if np.random.rand() > 0.5:
            img = TF.adjust_contrast(img, np.random.uniform(0.8, 1.2))
        return img

# Data loader setup
def get_data_loaders(batch_size=32):
    train_image_paths = glob("IDCIA_Augmentated_V2/images/train/*.tiff")
    val_image_paths = glob("IDCIA_Augmentated_V2/images/val/*.tiff")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        MicroscopyAugmentations(),
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

# Loss function with dynamic weighting
def weighted_loss(cell_count_pred, cell_count_true, location_loss, alpha=1.0):
    mae_loss = torch.abs(cell_count_pred - cell_count_true).mean()
    combined_loss = mae_loss + alpha * location_loss
    return combined_loss

# Model training
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, alpha=1.0, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.SmoothL1Loss()  # Robust loss for regression tasks
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, labels, cell_locations in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count_pred, predicted_locations = model(inputs)

            # Location loss
            location_loss = 0.0
            for i in range(inputs.size(0)):
                true_loc = cell_locations[i].to(device)
                num_locations = true_loc.size(0)
                pred_loc = predicted_locations[i].view(-1, 2)[:num_locations]
                dist_matrix = torch.cdist(pred_loc.unsqueeze(0), true_loc.unsqueeze(0)).squeeze(0)
                min_dists = dist_matrix.min(dim=1)[0]
                location_loss += min_dists.mean()

            location_loss /= inputs.size(0)
            loss = weighted_loss(cell_count_pred, labels, location_loss, alpha=alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Validation
        val_loss, val_mae = validate_model(model, val_loader, device, criterion)
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, Val MAE = {val_mae:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with Val Loss = {val_loss:.4f}")

    return model, train_losses, val_losses

# Validation
def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count_pred, _ = model(inputs)
            val_loss += criterion(cell_count_pred, labels).item()
            val_mae += torch.abs(cell_count_pred - labels).mean().item()
    return val_loss / len(val_loader), val_mae / len(val_loader)

# Main function
def main():
    train_loader, val_loader = get_data_loaders(batch_size=64)
    model = CellCounter(fine_tune=True)

    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=150, learning_rate=5e-4, alpha=1.0, weight_decay=1e-5
    )

    print("Training complete.")
    print(f"Best model saved to {output_dir}")

if __name__ == "__main__":
    main()
