"""
CIFAR10 classification problem

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %% library
import torch # pytorch lib, tensor
import torch.nn as nn # artificial neural network layers
import torch.optim as optim # optimizations algoritms module
import torchvision # image processing and pre-defined models
import torchvision.transforms as transforms # image transformations
import matplotlib.pyplot as plt # for visualization
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
def get_data_loaders(batch_size = 64): # every iterations data size
    
    transform = transforms.Compose([
        transforms.ToTensor(), # converts the image to tensor and 0-255
        transforms.Normalize(((0.5, 0.5, 0.5)), (0.5, 0.5, 0.5)) # normalize rgb channels
    ])
    
    # download cifar10 and create train, test
    train_set = torchvision.datasets.CIFAR10(root = "./data", train = True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root = "./data", train = False, download=True, transform=transform)
    
    #pytorch data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle= False)
    
    return train_loader, test_loader


# %% visualize dataset

def imshow(img):
    # before normalization 
    img = img / 2 + 0.5 # reverse normalization
    np_img = img.numpy() # tensor to numpy array
    plt.imshow(np.transpose(np_img, (1,2,0))) # displaying correctly for 3 RGB channel
    plt.show()
    
def get_sample_images(train_loader): # take sample images from data
    
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loaders()
    
    # n = number of samples 
    images, labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
    
# visualize(10)

# %% build CNN Model

class CNN(nn.Module):
    
    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1) # in = rgb 3, out = 32 filters, kernel_size 3x3 and edges 1
        self.relu = nn.ReLU() # ReLU activation function
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # 2x2 kernel, 2 step stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1) # in 32 outputs 64
        self.dropout = nn.Dropout(0.2) # %20 dropout
        self.fc1 = nn.Linear(64*8*8, 128) # filter = 64, fully connected layer = 4096, output = 128
        self.fc2 = nn.Linear(128, 10) # output layer
        
        # image 3x32x32 -> conv (32) -> relu (32) -> pool (16)
        # conv (16) -> relu (16) -> pool (8) -> image = 8x8
        
    def forward(self, x):
        """
            image 3x32x32 -> conv (32) -> relu (32) -> pool (16)
            conv (16) -> relu (16) -> pool (8) -> image = 8x8
            flatten
            fc1 -> relu -> dropout
            fc2 -> output
        """
        x = self.pool(self.relu(self.conv1(x))) # first convolution block
        x = self.pool(self.relu(self.conv2(x))) # second convolution block
        x = x.view(-1, 64*8*8) # flatten
        x = self.dropout(self.relu(self.fc1(x))) # fully connected layer
        x = self.fc2(x) # output layer
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# define loss function and optimizer
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # loss for multi class classification
    optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) # stochastic gradient descent
)

# %% training

def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    
    model.train() # set the model to training mode
    train_losses = [] # create a list to store loss values
    
    for epoch in range(epochs): # loop through the number of epochs
        total_loss = 0 # total loss
        for images, labels in train_loader: # Initialize total loss for this epoch
            images, labels = images.to(device), labels.to(device) # Move data to GPU/CPU
            
            optimizer.zero_grad() # Clear previous gradients
            outputs = model(images) # forward pass (predictions)
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # backward pass (compute gradients)
            optimizer.step() # learning Update weights
            
            total_loss += loss.item() # Add loss for this batch
        
        avg_loss = total_loss / len(train_loader) # calculate average loss for the epoch
        train_losses.append(avg_loss) # save it
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}") # print progress
        
    # plotting the Loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# train_loader, test_loader = get_data_loaders()
# model = CNN().to(device)
# criterion, optimizer = define_loss_and_optimizer(model)
# train_model(model, train_loader, criterion, optimizer, epochs = 10)

# %% test

def test_model(model, test_loader, dataset_type):
    
    model.eval() # evaluation mode
    correct = 0 # correct counter
    total = 0 # total correct counter
    
    with torch.no_grad(): # turn off gradient calculation
        for images, labels in test_loader: # evaluation using a test data set
            images, labels = images.to(device), labels.to(device) # move data to the device
             
            outputs = model(images) # prediction
            _, predicted = torch.max(outputs, 1) # select the highest probability class
            total += labels.size(0) # total number of data
            correct += (predicted == labels).sum().item() # count the correct guesses
            
    print(f"{dataset_type} accuracy: {100 * correct / total} %") # print accuracy

# test_model(model, test_loader, dataset_type = "test") # test accuracy: 63.21 %
# test_model(model, train_loader, dataset_type= "training") # training accuracy: 65.716 %

# https://paperswithcode.com/sota/image-classification-on-cifar-10


# %% main program

if __name__ == "__main__":
    
    # upload data
    train_loader, test_loader = get_data_loaders()
    
    # visualization
    visualize(10)
    
    # training
    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs = 10)
    
    # test
    test_model(model, test_loader, dataset_type = "test") # test accuracy: 62.5 %
    test_model(model, train_loader, dataset_type= "training") # training accuracy: 65.244 %
