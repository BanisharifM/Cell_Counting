# Use an official PyTorch image with CUDA for GPU support
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install necessary Python libraries
RUN pip install scipy numpy matplotlib torch torchvision tqdm

# Install DenseWeight and KDEpy
RUN pip install denseweight KDEpy

# Set up your working directory in the container
WORKDIR /workspace

# Copy all project files into the container
COPY . /workspace

# Run the training script
CMD ["python", "main7.py"]
