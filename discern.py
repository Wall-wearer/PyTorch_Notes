import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchsummary import summary

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x

# Load the trained model
model = torch.load("FashionModel.pkl")
model.eval()  # Set the model to evaluation mode

# Define the same data transformations
image_size = 28
data_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize to (28, 28)
    transforms.ToTensor()  # Convert to tensor with values normalized to [0, 1]
])

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = data_transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)


def predict(image_path):
    image = preprocess_image(image_path)
    output = model(image)
    prediction = torch.argmax(output, 1)
    return prediction.item()


def predict(image_path):
    image = preprocess_image(image_path)
    output = model(image)
    prediction = torch.argmax(output, 1).item()
    return prediction, class_names[prediction]


# Example usage
image_path = "Ankle boot.jpg"  # Replace with the path to your image
predicted_label = predict(image_path)
print(f"Predicted Label: {predicted_label}")

predicted_label, class_name = predict(image_path)
print(f"Predicted Label: {predicted_label}, Class Name: {class_name}")