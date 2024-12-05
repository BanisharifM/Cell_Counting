from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

train_transforms = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(root="IDCIA_Augmentated_V2/images/train", transform=train_transforms)
print(f"Number of images in train dataset: {len(dataset)}")
