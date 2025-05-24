"""
RNN: Recurrent Neural Networks: We used them in time series: short summary

Dataset selection

"""

# %% Create data and visualize
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_data(seq_length=50, num_samples=1000):
    """
        example: 3-element packet
        sequence: [2,3,4] # to store input sequences
        targets: [5] to store target values  
    """
    X = np.linspace(0, 100, num_samples)  # Generate num_samples amount of data between 0-100
    y = np.sin(X)
    sequence = []  # to store input sequences
    targets = []  # to store target values
    
    for i in range(len(X) - seq_length):
        sequence.append(y[i:i+seq_length])  # input
        targets.append(y[i + seq_length])  # the value following the input sequence
    
    # Visualize the data
    plt.figure(figsize=(8, 4))
    plt.plot(X, y, label='sin(t)', color='b', linewidth=2)
    plt.title('Sine Wave Graph')
    plt.xlabel('Time (radians)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return np.array(sequence), np.array(targets)
        
sequence, targets = generate_data()

# %% Create RNN model

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """            
            RNN -> Linear (output)
        """
        super(RNN, self).__init__()
        # input_size: input dimension
        # hidden_size: number of cells in the RNN hidden layer
        # num_layers: number of RNN layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN layer
        # output_size: output dimension or predicted value
        self.fc = nn.Linear(hidden_size, output_size)  # fully connected layer: output
        
    def forward(self, x):
        
        out, _ = self.rnn(x)  # pass input to RNN and get output
        out = self.fc(out[:, -1, :])  # take the output from the last time step and pass it to the fc layer
        return out
        
model = RNN(1, 16, 1, 1)
# %% RNN training

# Hyperparameters
seq_length = 50  # length of the input sequence
input_size = 1  # size of the input sequence
hidden_size = 16  # number of nodes in the RNN's hidden layer
output_size = 1  # size of the output or the predicted value
num_layers = 1  # number of RNN layers
epochs = 20  # number of times the model will be trained on the entire dataset
batch_size = 32  # number of samples used in each training step
learning_rate = 0.001  # learning rate for the optimization algorithm

# Prepare the data
X, y = generate_data(seq_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # convert to PyTorch tensor and add a dimension
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # convert to PyTorch tensor and add a dimension

dataset = torch.utils.data.TensorDataset(X, y)  # create PyTorch dataset
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # create data loader

# Define the model
model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # loss function: Mean Square Error - average squared error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimization = adaptive momentum

for epoch in range(epochs):
    for batch_x, batch_y in dataLoader:
        optimizer.zero_grad()  # zero the gradients
        pred_y = model(batch_x)  # get predictions from the model
        loss = criterion(pred_y, batch_y)  # compare model prediction with actual value and calculate loss    
        loss.backward()  # backpropagation to calculate gradients
        optimizer.step()  # update weights
    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# %% RNN test and evaluation

# Create test data
X_test = np.linspace(100, 110, seq_length).reshape(1, -1)  # first test data
y_test = np.sin(X_test)  # real value of the test data

X_test2 = np.linspace(120, 130, seq_length).reshape(1, -1)  # second test data
y_test2 = np.sin(X_test2)

# Convert from numpy to tensor
X_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
X_test2 = torch.tensor(y_test2, dtype=torch.float32).unsqueeze(-1)

# Make predictions using the model
model.eval()
prediction1 = model(X_test).detach().numpy()  # prediction for the first test data
prediction2 = model(X_test2).detach().numpy()

# Visualize the results
plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker="o", label="Training dataset")
plt.plot(X_test.numpy().flatten(), marker="o", label="Test 1")
plt.plot(X_test2.numpy().flatten(), marker="o", label="Test 2")

plt.plot(np.arange(seq_length, seq_length + 1), prediction1.flatten(), "ro", label="Prediction 1")
plt.plot(np.arange(seq_length, seq_length + 1), prediction2.flatten(), "ro", label="Prediction 2")
plt.legend()
plt.show()
