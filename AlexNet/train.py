import torch
import torch.nn as nn 

from model import AlexNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = AlexNet()

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum = 0.9
)

model.train()

num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    running_loss = 0.0

    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Forward Pass
        outputs = model(images)
        
        # Compute Loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        correct += (predicted == labels).sum().item() 

        # Clear old gradients 
        optimizer.zero_grad()

        # Compute new gradients 
        loss.backward()

        # Update parameters 
        optimizer.step()

        total += len(images)

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

    accuracy = 100 * (correct / total)
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {accuracy}")

    torch.save(model.state_dict(), "alexnet.pth")


