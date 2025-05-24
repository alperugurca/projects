import torch  # Import PyTorch library
import torch.nn as nn  # Import PyTorch module for neural network layers
import torch.optim as optim  # Import optimization algorithms
import torchvision.transforms as transforms  # Import module for image transformations
import torchvision.datasets as datasets  # Import module to use prebuilt datasets
import torchvision.models as models  # Import module to load pretrained models
from torch.utils.data import DataLoader  # Import PyTorch module for data loading
from tqdm import tqdm  # Import progress bar to monitor training
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Device selection (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset transformations (Data Augmentation added)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for MobileNet input
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip for data augmentation
    transforms.RandomRotation(10),  # Apply random rotation up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color variations
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize input for test dataset
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Load Oxford Flowers 102 dataset (train and test splits)
train_dataset = datasets.Flowers102(root="./data", split="train", transform=transform_train, download=True)  # Download training dataset
test_dataset = datasets.Flowers102(root="./data", split="val", transform=transform_test, download=True)  # Download test dataset

# Select 5 random samples
indices = torch.randint(len(train_dataset), (5,))
samples = [train_dataset[i] for i in indices]

# Visualization
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (image, label) in enumerate(samples):
    image = image.numpy().transpose((1, 2, 0))  # Convert tensor to image format
    image = (image * 0.5) + 0.5  # Undo normalization
    axes[i].imshow(image)
    axes[i].set_title(f"Class: {label}")
    axes[i].axis("off")
plt.show()

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Create training data loader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # Create test data loader

# Load MobileNetV2 model (pretrained weights)
model = models.mobilenet_v2(pretrained=True)  # Download MobileNetV2 model

# Replace classifier layer (adapt output to 102 classes)
num_ftrs = model.classifier[1].in_features  # Get input features of existing classifier layer
model.classifier[1] = nn.Linear(num_ftrs, 102)  # Replace final layer with one for 102 flower classes
model = model.to(device)  # Move model to selected device (GPU/CPU)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)  # Use Adam optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Add StepLR for learning rate scheduling

# Training the model
epochs = 3  # Set training loop to 3 epochs
for epoch in tqdm(range(epochs)):  # Show progress bar for each epoch
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize cumulative loss
    for images, labels in tqdm(train_loader):  # Loop over training data
        images, labels = images.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()  # Zero previous gradients
        outputs = model(images)  # Make predictions
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update total loss
    scheduler.step()  # Update learning rate
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")  # Print average loss at epoch end

# Save model
torch.save(model.state_dict(), "mobilenet_flowers102.pth")

# %% Test and evaluate

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds) 
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(all_labels, all_preds))
