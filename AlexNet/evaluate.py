import torch
import torch.nn as nn

from model import AlexNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = AlexNet()
state_dict = torch.load("alexnet.pth")
model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

model.eval()

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

        test_correct += (predicted == labels).sum().item()
        test_total += len(images)

accuracy = 100 * (test_correct / test_total)
print(f"Accuracy: {accuracy}" )






