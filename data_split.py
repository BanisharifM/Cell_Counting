import os
import glob
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Define directories
image_dir = "IDCIA/images"
train_dir = "IDCIA/train"
val_dir = "IDCIA/val"

# Create train and val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all TIFF images in the image_dir
image_paths = glob.glob(os.path.join(image_dir, "*.tiff"))

# Shuffle the image paths to ensure randomness
random.shuffle(image_paths)

# Calculate the number of images for validation (20%)
val_size = int(0.2 * len(image_paths))

# Split the dataset
val_images = image_paths[:val_size]
train_images = image_paths[val_size:]

# Function to copy images to the respective directories
def copy_images(image_list, destination):
    for image_path in image_list:
        shutil.copy(image_path, destination)

# Copy validation images to the val folder
copy_images(val_images, val_dir)

# Copy training images to the train folder
copy_images(train_images, train_dir)

print(f"Total images: {len(image_paths)}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
