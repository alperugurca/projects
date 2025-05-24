# https://archive.ics.uci.edu/dataset/53/iris

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% loading the dataset
# classification problem: iris dataset is a classification problem with 3 different classes

df = pd.read_csv("iris.data", header=None)

X = df.iloc[:, :-1].values  # assign the first 4 columns to variable X
y, _ = pd.factorize(df.iloc[:, -1])  # encode class labels as integers

# standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def to_tensor(data, target):
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)

# %% defining the RBFN model and rbf_kernel

def rbf_kernel(X, centers, beta):
    return torch.exp(-beta * torch.cdist(X, centers)**2)

class RBFN(nn.Module):
    
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))  # randomly initialize RBF centers
        self.beta = nn.Parameter(torch.ones(1) * 2.0)  # beta controls the width of the RBF
        self.linear = nn.Linear(num_centers, output_dim)  # map output to a fully connected layer
    
    def forward(self, x):  # forward pass
        # compute the RBF kernel function
        phi = rbf_kernel(x, self.centers, self.beta)
        return self.linear(phi)

# model = RBFN(4, 10, 3)
# %% model training

num_centers = 10
model = RBFN(input_dim=4, num_centers=num_centers, output_dim=3)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # reset gradients
    outputs = model(X_train)  # prediction, i.e., forward pass
    loss = criterion(outputs, y_train)  # compute loss
    loss.backward()  # backpropagation
    optimizer.step()  # update weights
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# %% test and evaluation

with torch.no_grad():
    y_pred = model(X_test)  # make predictions on test data
    accuracy = (torch.argmax(y_pred, axis=1) == y_test).float().mean().item()  # calculate accuracy
    print(f"accuracy: {accuracy}")