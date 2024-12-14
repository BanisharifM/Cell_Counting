import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from glob import glob
from tqdm import tqdm
from dataset_handler import CellDataset
from model import CellCounter
import numpy as np
import matplotlib.pyplot as plt

# Set the output directory for saving models and plots
output_dir = "Experiments/67"
os.makedirs(output_dir, exist_ok=True)


# Custom collate function
def custom_collate_fn(batch):
    images, labels, cell_locations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    padded_cell_locations = pad_sequence(cell_locations, batch_first=True, padding_value=-1)
    return images, labels, padded_cell_locations


# Data augmentation class
class MicroscopyAugmentations:
    def __call__(self, img):
        if np.random.rand() > 0.5:
            img = transforms.functional.hflip(img)
        if np.random.rand() > 0.5:
            img = transforms.functional.vflip(img)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30)
            img = transforms.functional.rotate(img, angle)
        if np.random.rand() > 0.5:
            img = transforms.functional.adjust_brightness(img, np.random.uniform(0.8, 1.2))
        if np.random.rand() > 0.5:
            img = transforms.functional.adjust_contrast(img, np.random.uniform(0.8, 1.2))
        return img


# Data loader function with oversampling for high cell counts
def get_data_loaders(batch_size=32):
    # Paths to images
    train_image_paths = glob("IDCIA_Augmentated_V2/images/train/*.tiff")
    val_image_paths = glob("IDCIA_Augmentated_V2/images/val/*.tiff")

    # Transforms
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

    # Datasets
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)

    # Extract labels for weighted sampling
    train_labels = [label for _, label, _ in train_dataset]  # Assuming CellDataset returns (image, label, locations)

    # Weighted sampling
    weights = [1.0 if label > 500 else 0.5 for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    return train_loader, val_loader


# Weighted MAE loss
def weighted_mae_loss(pred_count, true_count):
    weights = true_count / (true_count.mean() + 1e-8)
    loss = (torch.abs(pred_count - true_count) * weights).mean()
    return loss


# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
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

            # Weighted MAE loss
            count_loss = weighted_mae_loss(cell_count_pred, labels)
            loss = count_loss + 0.1 * location_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        val_loss, val_mae = validate_model(model, val_loader, device)
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, Val MAE = {val_mae:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with Val Loss = {val_loss:.4f}")

    return model, train_losses, val_losses


# Validation function
def validate_model(model, val_loader, device):
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1)
            cell_count_pred, _ = model(inputs)
            val_loss += F.l1_loss(cell_count_pred, labels).item()
            val_mae += torch.abs(cell_count_pred - labels).mean().item()
    return val_loss / len(val_loader), val_mae / len(val_loader)


# Main function
def main():
    train_loader, val_loader = get_data_loaders(batch_size=64)
    model = CellCounter(fine_tune=True)

    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=150, learning_rate=1e-3, weight_decay=1e-4
    )

    print("Training complete.")
    print(f"Best model saved to {output_dir}")


if __name__ == "__main__":
    main()
