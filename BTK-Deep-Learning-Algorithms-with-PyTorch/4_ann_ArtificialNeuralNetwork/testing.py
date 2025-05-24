"""
Test
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Prediction function
def predict_custom_image(image_path, model):
    image = Image.open(image_path).convert("L")  # grayscale
    image = ImageOps.invert(image)  # if background is black
    image = image.resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    plt.imshow(image, cmap="gray")
    plt.title(f"Predicted: {predicted.item()}")
    plt.axis("off")
    plt.show()

# Load model from state_dict
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("mnist_ann.pth", map_location=device))
model.eval()

# Predict (update with your actual image path)
predict_custom_image(r"C:\Users\AE\Desktop\Deep Learning Algorithms with PyTorch\4\my_digit.png", model)