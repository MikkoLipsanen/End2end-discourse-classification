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

parser.add_argument('--data_path', type=str, default='data/samples/', help='file containing data')
parser.add_argument('--res_path', type=str, default='results/attn_results.pkl', help='path to save results')
parser.add_argument('--val_size', type=float, default=0.2, help='defines the size of the validation dataset (as portion of the training data)')
parser.add_argument('--batch_size', type=int, default=100, help='defines batch size')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--lr', type=float, default=0.2, help='defines the learning rate')
parser.add_argument('--epochs', type=int, default=10, help='defines the number of epochs')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=2, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.3, help='dropout probability for TransformerEncoder')


args = parser.parse_args()
print(args)

##############Load input data############

# Load the training data 
tr_dict = torch.load(args.data_path + "tr_data.pt")
# Get input data and labels 
data_tensor = tr_dict['data']
labels = tr_dict['labels']
# Get the maximum amount of documents per timepoint for padding
max_docs = tr_dict['max_size']

del tr_dict

# Get the training and validation data and labels
val_size = int(round(args.val_size * len(labels), -2))
tr_size = len(labels) - val_size

print('Validation data size: ', val_size)
print('Train data size: ', tr_size)

tr_data = data_tensor[:,:tr_size,:]
tr_labels = labels[:tr_size]

val_data = data_tensor[:,tr_size:,:]
val_labels = labels[tr_size:]

del data_tensor
del labels

# Load the test data 
ts_dict = torch.load(args.data_path + "ts_data.pt")
# Get input data and labels 
ts_data = ts_dict['data']
ts_labels = ts_dict['labels']

print('Test data size: ', len(ts_labels))


#########Initialize dataset and dataloader#######

# Creates a pytorch dataset
class EmbeddingsDataset(Dataset):
    def __init__(self, x, y, max_docs):
        self.x = x
        self.y = y
        self.length = self.y.shape[0]
        self.max_docs = max_docs
    
    def __getitem__(self,idx):
        
        timepoint = self.x[:,idx,:]
        
        # Adds padding if required
        if timepoint.shape[0] < self.max_docs:
            pad = self.max_docs - timepoint.shape[0]
            timepoint = F.pad(input=timepoint, pad=(0, 0, 0, pad), mode='constant', value=0)
            
        return timepoint, self.y[idx]
    
    def __len__(self):
        
        return self.length

# Creates train, validation and test datasets
train_data = EmbeddingsDataset(tr_data, tr_labels, max_docs)
validation_data = EmbeddingsDataset(val_data, val_labels, max_docs)
test_data = EmbeddingsDataset(ts_data, ts_labels, max_docs)

# Creates dataloaders for train, validation and test data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Print the batch shape
train_features, train_labels = next(iter(train_dataloader))
print(f"Train batch shape: {train_features.size()}")


##############Define the network modules############

# Define the MLP network class
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        
        # define network layers      
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 200*50)
        self.fc2 = nn.Linear(200*50, 200*20)
        self.fc3 = nn.Linear(200*20, 200*20)
        self.fc4 = nn.Linear(200*20, 50*10)
        self.fc5 = nn.Linear(50*10, output_size)

    def forward(self, x):
        # define forward pass
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
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
mlp_input_size = max_docs*args.emb_size
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


#######Define the train, validation and test iteration functions#######

# Function for training the model
def train_model():
    
    # Set the networks to train mode
    mlp.train()
    attn.train()
    
    loss = 0

    num_batches = len(train_dataloader)
    
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
    train_loss = loss / num_batches
    
    return train_loss

# Function for evaluating the model
def eval_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    mlp.eval()
    attn.eval()
    
    loss = 0
    correct = 0
    
    size = len(validation_dataloader.dataset)
    num_batches = len(validation_dataloader)
    
    # Gradients are not calculated during evaluation
    with torch.no_grad():
        
        # Loop over test data batches
        for (x_val, y_val) in validation_dataloader:
            
            # Calculate output for test data 
            mlp_pred = mlp(x_val)
            attn_pred = attn(mlp_pred.unsqueeze(1))
        
            # Calculate loss for test data
            val_loss = criterion(attn_pred, y_val)
            loss += val_loss.item()
            
            # Calculate the number of correct predictions in batch  
            correct += (torch.sum(torch.round(attn_pred.reshape(-1).detach()) == y_val.detach())).item()
        
        # Calculate average loss and accuracy for the epoch
        validation_loss = loss / num_batches
        validation_accuracy = correct / size
    
    return validation_loss, validation_accuracy

# Function for testing the model
def test_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    mlp.eval()
    attn.eval()
    
    loss = 0
    correct = 0
    
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    
    # Gradients are not calculated during testing
    with torch.no_grad():
        
        # Loop over test data batches
        for (x_test, y_test) in test_dataloader:
            
            # Calculate output for test data 
            mlp_pred = mlp(x_test)
            attn_pred = attn(mlp_pred.unsqueeze(1))
        
            # Calculate loss for test data
            test_loss = criterion(attn_pred, y_test)
            loss += test_loss.item()
            
            # Calculate accuracy (portion of labels predicted correctly)
            correct += (torch.sum(torch.round(attn_pred.reshape(-1).detach()) == y_test.detach())).item()
        
        # Calculate average loss and accuracy for the epoch
        test_loss = loss / num_batches
        test_accuracy = correct / size
        
    return test_loss, test_accuracy


#########Train the model##########

# Save loss and accuracy for each epoch
tr_losses = []
val_losses = []
val_accuracies = []

# Calculate training and evaluation loss and accuracy for each epoch
for epoch in range(1, args.epochs+1): 
    tr_loss = train_model()
    val_loss, accuracy = eval_model()
    tr_losses.append(tr_loss)
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)
    
    print("|Epoch %d | Training loss : %.3f | Validation loss %.3f | Accuracy %.3f"%(epoch, tr_loss, val_loss, accuracy))

print("Mean validation accuracy: %.3f" %np.mean(val_accuracies))


##########Get the model accuracy for the test set#########

test_loss, test_accuracy = test_model()

print("Test loss : %.3f | Test accuracy %.3f" %(test_loss, test_accuracy))


##########Save the results#################

results = {'tr_loss': tr_losses, 'val_loss': val_losses, 'accuracy': val_accuracies}

# Save loss and accuracy
file_to_write = open(args.res_path, 'wb')
pickle.dump(results, file_to_write)