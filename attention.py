import pickle
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# Uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# python attention.py --epochs 10   

parser = argparse.ArgumentParser(description='Attention For Discourse Detection')

parser.add_argument('--data_path', type=str, default='data/samples/tensors_5.pt', help='file containing data')
parser.add_argument('--res_path', type=str, default='results/attn_results_5.pkl', help='path to save results')
parser.add_argument('--tr_size', type=float, default=0.8, help='defines the size of the training set')
parser.add_argument('--batch_size', type=int, default=100, help='defines batch size')
parser.add_argument('--lr', type=float, default=0.2, help='defines the learning rate')
parser.add_argument('--epochs', type=int, default=10, help='defines the number of epochs')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=2, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='dropout probability for TransformerEncoder')


args = parser.parse_args()
print(args)

# Load the data dict 
tensor_dict = torch.load(args.data_path)

# Get input data and labels 
data_tensor = tensor_dict['data']
labels = tensor_dict['labels']

del tensor_dict

# Split data into train and test sets so that the order of timepoints is preserved
train_size = int(round(args.tr_size * len(labels), -2))
test_size = len(labels) - train_size

print('Train data size: ', train_size)
print('Test data size: ', test_size)

train_data = data_tensor[:,:train_size,:]
train_labels = labels[:train_size]

test_data = data_tensor[:,train_size:,:]
test_labels = labels[train_size:]

# Creates a pytorch dataset
class EmbeddingsDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.length = self.y.shape[0]
    
    def __getitem__(self,idx):
        return self.x[:,idx,:], self.y[idx]
    
    def __len__(self):
        return self.length

# Creates train and test datasets
tr_data = EmbeddingsDataset(train_data, train_labels)
ts_data = EmbeddingsDataset(test_data, test_labels)

# Creates dataloaders for train and test data
train_dataloader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(ts_data, batch_size=args.batch_size, shuffle=True)

# Print the batch shape
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")

# Define the MLP network class
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        
        # define network layers      
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 200*50)
        self.fc2 = nn.Linear(200*50, 200*20)
        self.fc3 = nn.Linear(200*20, 50*10)
        self.fc4 = nn.Linear(50*10, output_size)

    def forward(self, x):
        # define forward pass
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x

# Define the Attention network class
class ATTN(nn.Module):
    def __init__(self, features, heads, layers, dropout):
        super(ATTN, self).__init__()

        # define network layers  
        encoder_layer = nn.TransformerEncoderLayer(d_model=features, nhead=heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(features*features, features)
        self.sigmoid = nn.Sigmoid()
        
    # Define forward pass
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(torch.flatten(x))
        x = self.sigmoid(x)
        
        return x

# Define input and output size for MLP
mlp_input_size = data_tensor.size()[0]*data_tensor.size()[2]
mlp_output_size = 100

# Instantiate the models
mlp = MLP(mlp_input_size, mlp_output_size).to(device)
attn = ATTN(mlp_output_size, args.n_heads, args.attn_layers, args.attn_dropout).to(device)

# Print model architectures
print('MLP: ', mlp)
print('Attention: ', attn)

# Define optimizer and loss function
optimizer = torch.optim.SGD(list(mlp.parameters()) + list(attn.parameters()), lr=args.lr)
criterion = nn.BCELoss()

# Function for training the model
def train_model():
    
    # Set the networks to train mode
    mlp.train()
    attn.train()
    
    loss = 0
    
    # Loop over train data batches
    for (x_train, y_train) in train_dataloader:

        # Calculate output
        mlp_output = mlp(x_train)
        attn_output = attn(mlp_output.unsqueeze(1))

        # Calculate training loss
        tr_loss = criterion(attn_output, y_train)
        loss += tr_loss.item()
        
        # Backpropagate loss
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()
    
    # Calculate average loss for the epoch
    train_loss = loss / len(train_dataloader)
    
    return train_loss

# Function for evaluating the model
def eval_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    mlp.eval()
    attn.eval()
    
    loss = 0
    accuracy = 0
    
    # Gradients are not calculated during evaluation
    with torch.no_grad():
        
        # Loop over test data batches
        for (x_test, y_test) in test_dataloader:
            
            # Calculate output for test data 
            mlp_pred = mlp(x_test)
            attn_pred = attn(mlp_pred.unsqueeze(1))
        
            # Calculate loss for test data
            val_loss = criterion(attn_pred, y_test)
            loss += val_loss.item()
            
            # Calculate accuracy (portion of labels predicted correctly)
            acc = (torch.sum(torch.round(attn_pred.reshape(-1).detach()) == y_test.detach()) / len(y_test)).item()
             
            accuracy += acc
        
        # Calculate average loss and accuracy for the epoch
        validation_loss = loss / len(test_dataloader)
        validation_accuracy = accuracy / len(test_dataloader)
    
    return validation_loss, validation_accuracy

# Save loss and accuracy for each epoch
tr_losses = []
val_losses = []
accuracies = []

# Calculate training and evaluation loss and accuracy for each epoch
for epoch in range(1, args.epochs+1): 
    tr_loss = train_model()
    val_loss, accuracy = eval_model()
    tr_losses.append(tr_loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)
    
    print("|Epoch %d | Training loss : %.3f | Validation loss %.3f | Accuracy %.3f"%(epoch, tr_loss, val_loss, accuracy))

print("Mean accuracy: %.3f"%np.mean(accuracies))

results = {'tr_loss': tr_losses, 'val_loss': val_losses, 'accuracy': accuracies}

# Save loss and accuracy
file_to_write = open(args.res_path, 'wb')
pickle.dump(results, file_to_write)