import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import os

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipped corrupt image: {path}")
            return self.__getitem__((index + 1) % len(self))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Datasets and loaders
train_dataset = SafeImageFolder('data/train', transform=train_transform)
val_dataset = SafeImageFolder('data/val', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

best_acc=0.0

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")
    val_acc=evaluate(model, val_loader)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, 'models/fake_art_classifier_full.pt')
        print(f"✅ New best model saved at {best_acc:.2f}% accuracy")


# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model, 'models/fake_art_classifier_full.pt')
