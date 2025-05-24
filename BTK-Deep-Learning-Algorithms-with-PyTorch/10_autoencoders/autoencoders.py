"""
Problem definition: Data compression -> Autoencoders
Data: FashionMNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

# %% Load and preprocess the dataset
transform = transforms.Compose([transforms.ToTensor()])  # convert images to tensor and normalize to [0-1]

# Download and load training and test datasets
train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Batch size
batch_size = 128

# Create training and test data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% Build autoencoders

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),               # 28x28 (2D) -> 784 (1D) vector
            nn.Linear(28*28, 256),      # fully connected layer: 784 -> 256
            nn.ReLU(),                  # activation function
            nn.Linear(256, 64),         # fully connected layer: 256 -> 64
            nn.ReLU()
        )
                
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),         # fully connected layer: 64 -> 256
            nn.ReLU(),                  # activation function
            nn.Linear(256, 28*28),      # fully connected layer: 256 -> 784
            nn.Sigmoid(),               # sigmoid used to keep values in range [0-1]
            nn.Unflatten(1, (1, 28, 28)) # reshape the 1D output back to 28x28
        )
            
    def forward(self, x):
        encoded = self.encoder(x)      # encode the input
        decoded = self.decoder(encoded) # decode to reconstruct image
        return decoded

# %% Callback: early stopping

class EarlyStopping:  # early stopping (custom callback class)

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience       # number of epochs with no improvement after which training will be stopped
        self.min_delta = min_delta     # minimum change in the monitored quantity to qualify as an improvement
        
        self.best_loss = None          # best loss seen so far
        self.counter = 0               # counter for epochs without improvement
        
    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:  # improvement detected
            self.best_loss = loss
            self.counter = 0           # reset counter
        else:                          # no improvement
            self.counter += 1
            
        if self.counter >= self.patience:  # if no improvement for 'patience' epochs, stop training
            return True
        
        return False

# %% Model training

# Hyperparameters
epochs = 50            # number of training epochs
learning_rate = 1e-3   # learning rate

# Define model, loss and optimizer
model = AutoEncoder()
criterion = nn.MSELoss()                             # loss function: mean squared error
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # optimizer
early_stopping = EarlyStopping(patience=3, min_delta=0.001)   # early stopping object

# Training function
def training(model, train_loader, optimizer, criterion, early_stopping, epochs):
    model.train()  # set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in train_loader:
            optimizer.zero_grad()           # reset gradients
            outputs = model(inputs)         # forward pass
            loss = criterion(outputs, inputs)  # compute loss between input and reconstructed output
            loss.backward()                 # backpropagation
            optimizer.step()                # update weights
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)  # average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.5f}")
        
        # Early stopping
        if early_stopping(avg_loss):  # check if early stopping criteria met
            print(f"Early stopping at epoch {epoch+1}")
            break

training(model, train_loader, optimizer, criterion, early_stopping, epochs)

# %% Model testing

from scipy.ndimage import gaussian_filter

def compute_ssim(img1, img2, sigma=1.5):
    """
    Computes similarity between two images using SSIM
    """
    C1 = (0.01*255)**2  # SSIM constant
    C2 = (0.03*255)**2  # SSIM constant
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Compute means
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(img1**2, sigma) - mu1_sq  # variance
    sigma2_sq = gaussian_filter(img2**2, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1*img2, sigma) - mu1_mu2  # covariance
    
    # Compute SSIM map
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def evaluate(model, test_loader, n_images=10):
    model.eval()  # set model to evaluation mode
    
    with torch.no_grad():  # disable gradient calculation
        for batch in test_loader: 
            inputs, _ = batch
            outputs = model(inputs)  # reconstruct the images
            break
    
    inputs = inputs.numpy()
    outputs = outputs.numpy()
    
    fig, axes = plt.subplots(2, n_images, figsize=(n_images, 3))  # for visualization
    ssim_scores = []  # to store SSIM scores
    
    for i in range(n_images):
        img1 = np.squeeze(inputs[i])     # original image
        img2 = np.squeeze(outputs[i])    # reconstructed image
        
        ssim_score = compute_ssim(img1, img2)  # compute similarity
        ssim_scores.append(ssim_score)
        
        axes[0, i].imshow(img1, cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(img2, cmap="gray")
        axes[1, i].axis("off")
        
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Decoded image")
    plt.show()
    
    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")

evaluate(model, test_loader, n_images=10)
