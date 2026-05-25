import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#provaGit
# device GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trasformazione immagini
transform = transforms.ToTensor()

# dataset MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# dataloader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# mini CNN
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3
        )

        self.relu = nn.ReLU()

        self.fc = nn.Linear(8 * 26 * 26, 10)

    def forward(self, x):

        x = self.conv(x)

        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# modello
model = SimpleCNN().to(device)

# prende un batch
images, labels = next(iter(train_loader))

# sposta su GPU
images = images.to(device)

# forward
outputs = model(images)

print("Shape input:")
print(images.shape)

print("\nShape output:")
print(outputs.shape)

# mostra immagine
plt.imshow(images[0].cpu().squeeze(), cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()