"""
text generation with lstm
can remember the distant past
"""

# lstm reminder

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter  # to calculate word frequencies
from itertools import product  # to create combinations for grid search

# %% data loading and preprocessing
# product reviews
text = """This product exceeded my expectations.  
The material quality is really good.  
Shipping was fast and arrived without issues.  
Great performance for its price.  
I definitely recommend and suggest it!"""

# data preprocessing:
# remove punctuation marks,
# convert to lowercase
# split into words

words = text.replace(".", "").replace("!","").lower().split()

# calculate word frequencies and create indexing
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse = True)  # sort by frequency descending
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# prepare training data
data = [(words[i], words[i+1]) for i in range(len(words) - 1)]

# %% define lstm model

class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()  # call the parent constructor
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):  # forward pass function
        """
            input -> embedding -> lstm -> fc -> output
        """
        x = self.embedding(x)  # input -> embedding
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output = self.fc(lstm_out.view(1,-1))
        return output 

model = LSTM(len(vocab), embedding_dim=8, hidden_dim=32)

# %% hyperparameter tuning

# word list -> tensor
def prepare_squence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype = torch.long)

# define hyperparameter tuning combinations
embedding_sizes = [8, 16]  # embedding dimensions to try
hidden_sizes = [32, 64]  # hidden layer sizes to try
learning_rates = [0.01, 0.005]  # learning rates

best_loss = float("inf")  # to keep track of the lowest loss
best_params = {}  # to store the best parameters

print("Starting hyperparameter tuning...")

# grid search
for emb_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    print(f"Trial: Embedding: {emb_size}, Hidden: {hidden_size}, learning_rate: {lr}")
    
    # define the model
    model = LSTM(len(vocab), emb_size, hidden_size)  # create model with selected params
    loss_function = nn.CrossEntropyLoss()  # entropy loss function
    optimizer = optim.Adam(model.parameters(), lr = lr)  # Adam optimizer with selected lr

    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0  # reset loss at the beginning of each epoch
        for word, next_word in data: 
            model.zero_grad()  # zero gradients
            input_tensor = prepare_squence([word], word_to_ix)  # convert input to tensor
            target_tensor = prepare_squence([next_word], word_to_ix)  # convert target to tensor
            output = model(input_tensor)  # prediction
            loss = loss_function(output, target_tensor)
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.5f}")
        total_loss = epoch_loss 
    
    # save the best model
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_dim": emb_size, "hidden_dim":hidden_size, "learning_rate":lr}
    print()

print(f"Best params: {best_params}")

# %% lstm training

final_model = LSTM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'])
optimizer = optim.Adam(final_model.parameters(), lr = best_params['learning_rate'])
loss_function = nn.CrossEntropyLoss()  # entropy loss function

print("Final model training")
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0 
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_squence([word], word_to_ix)
        target_tensor = prepare_squence([next_word], word_to_ix)
        output = final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Final Model Epoch: {epoch}, Loss: {epoch_loss:.5f}")

# %% testing and evaluation

# word prediction function: generates n words starting from a given word
def predict_sequence(start_word, num_words):
    current_word = start_word  # set current word to starting word
    output_sequence = [current_word]  # output sequence
    
    for _ in range(num_words):  # predict specified number of words
        with torch.no_grad():  # no gradient computation
            input_tensor = prepare_squence([current_word], word_to_ix)  # word -> tensor
            output = final_model(input_tensor)
            predicted_idx = torch.argmax(output).item()  # index of word with highest probability
            predicted_word = ix_to_word[predicted_idx]  # return word corresponding to index
            output_sequence.append(predicted_word)
            current_word = predicted_word  # update current word for next prediction
    return output_sequence  # return predicted word sequence
       
"""
Original Text:
This product exceeded my expectations.  
The material quality is really good.  
Shipping was fast and arrived without issues.  
Great performance for its price.  
I definitely recommend and suggest it!
"""     
start_word = "and"
num_predictions = 10
predicted_sequence = predict_sequence(start_word, num_predictions)
print(" ".join(predicted_sequence)) 
