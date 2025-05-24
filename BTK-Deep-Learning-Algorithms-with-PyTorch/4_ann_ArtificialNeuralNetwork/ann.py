"""
Number Classification with MNIST dataset
MNIST
ANN: Artificial Neural Network

"""
# %% library
import torch # pytorch lib, tensor
import torch.nn as nn # artificial neural network layers
import torch.optim as optim # optimizations algoritms module
import torchvision # image processing and pre-defined models
import torchvision.transforms as transforms # image transformations
import matplotlib.pyplot as plt # for visualization

# optional: detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
def get_data_loaders(batch_size = 64): # every iterations data size
    
    transform = transforms.Compose([
        transforms.ToTensor(), # converts the image to tensor and 0-255
        transforms.Normalize((0.5,), (0.5,)) # pixel values -1 to 1    
    ])
    
    # download mnist and create train, test
    train_set = torchvision.datasets.MNIST(root = "./data", train=True, download=True, transform = transform)
    test_set = torchvision.datasets.MNIST(root = "./data", train=False, download=True, transform = transform)
    
    #pytorch data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle= False)
    
    return train_loader, test_loader


# train_loader, test_loader = get_data_loaders()

# data visualization
def visualize_samples(loader, n):
    images,labels = next(iter(loader)) # take first batch visuals and labels
    print(images[0].shape)
    fig, axes = plt.subplots(1, n, figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off") # dont show axis
    plt.show()
    
#visualize_samples(train_loader, 4)



# %% define ann model

class NeuralNetwork(nn.Module): # inherits from nn.module class

    def __init__(self): # nn components for build
        super(NeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten() # convert images (2d) into vectors (1d)
        
        self.fc1 = nn.Linear(28*28, 128) # first fully connected layer input = 28x28=784, output=128
        
        self.relu = nn.ReLU() # activation function
        
        self.fc2 = nn.Linear(128, 64) # second fully connected layer: input size = 128, output size = 64 
        
        self.fc3 = nn.Linear(64, 10) # cikti katmani olustur: input size = 64 , output size = 10(0-9 etiketleri)
        
    def forward(self, x): # forward propagation: input = x
        
        x = self.flatten(x) # for initial, flatten 784 vectors
        x = self.fc1(x) # first connected layer
        x = self.relu(x) # activation function
        x = self.fc2(x) # second connected layer
        x = self.relu(x) # activation function
        x = self.fc3(x) # output layer
        
        return x # return the output


# create model and compile
#model = NeuralNetwork().to(device)

# choose loss func and optimization algoritm
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # multi class classification problems loss function
    optim.Adam(model.parameters(), lr = 0.001) # update weights with adam
)

# criterion, optimizer = define_loss_and_optimizer(model)

# %% train
def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    
    model.train() # change to trainig mode
    train_losses = [] # each loss values list
    
    for epoch in range(epochs): # epochs = total training steps
        total_loss = 0 # total loss value
    
        for images, labels in train_loader: # iterate over all training data
            images, labels = images.to(device), labels.to(device) # move data to device
            
            optimizer.zero_grad() # reset gradients
            predictions = model(images) # appy model forward propagation
            loss = criterion(predictions, labels) # calculate loss, use y_pred and y_real
            loss.backward() # back propagation
            optimizer.step() # update weights
            
            total_loss = total_loss + loss.item()
        
        avg_loss = total_loss / len(train_loader) # average loss 
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")
    
    # loss graph
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# train_model(model, train_loader, criterion, optimizer, epochs=5)


# %% test
def test_model(model, test_loader):
    model.eval() # change to evaluation mode
    correct = 0 # correct data counter
    total = 0 # total data counter
    
    with torch.no_grad(): # turn off gradient calc
        for images, labels in test_loader: # iterate all test data
            images, labels = images.to(device), labels.to(device) # move data to device
            predictions = model(images)
            _, predicted = torch.max(predictions, 1) # highest probability class
            total += labels.size(0) # update total data count
            correct += (predicted == labels).sum().item() # count correct predictions
    
    print(f"Test Accuracy: {100*correct/total:.3f}%")
    
# test_model(model, test_loader)

# %% main

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader, 5)
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)
    
    
    
# %% save
# Save the model weights
torch.save(model.state_dict(), "mnist_ann.pth")
print("Model saved as mnist_ann.pth")
    
    
    
    
    