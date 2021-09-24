import pickle
import numpy as np
import argparse
import time
import random
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# Uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# python mix.py --tr_size 800 --val_size 200 --ts_size 200 --continue_tr no --epochs 10 --attn_layers 4

parser = argparse.ArgumentParser(description='NN For Discourse Detection')

parser.add_argument('--data_path', type=str, default='samples/', help='file containing data')
parser.add_argument('--res_path', type=str, default='results/', help='path to save results')
parser.add_argument('--model_path', type=str, default='model/triple_model.pt', help='path to save the model')
parser.add_argument('--continue_tr', type=str, default='no', help='continue training the saved model')
parser.add_argument('--batch_size', type=int, default=200, help='defines batch size')
parser.add_argument('--timepoints', type=int, default=100, help='defines number of timepoints in sample set')
parser.add_argument('--tr_size', type=int, default=4, help='defines the size of the training dataset')
parser.add_argument('--val_size', type=int, default=2, help='defines the size of the validation dataset')
parser.add_argument('--ts_size', type=int, default=2, help='defines the size of the test dataset')
parser.add_argument('--lr', type=float, default=0.1, help='defines the learning rate')
parser.add_argument('--epochs', type=int, default=6, help='defines the number of epochs')
parser.add_argument('--hidden_size', type=int, default=10, help='bLSTM hidden state size')
parser.add_argument('--lstm_layers', type=int, default=2, help='number of bLSTM layers')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--lstm_dropout', type=float, default=0.2, help='dropout probability for bLSTM')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=1, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='dropout probability for TransformerEncoder')


args = parser.parse_args()
print(args)

############Create train, validation and test splits#########

# Load the number indicating max amount of docs per timepoint
with open(args.data_path + "max_docs_triple.txt") as f:
    max_docs = f.read()
    
max_docs = int(max_docs)

# List all files in train and test folders
tr_val = [f for f in listdir(args.data_path + "train" + "/triplets") if isfile(join(args.data_path + "train" + "/triplets", f))]
test = [f for f in listdir(args.data_path + "test" + "/triplets") if isfile(join(args.data_path + "test" + "/triplets", f))]

# Shuffle the order of files in the list
random.shuffle(tr_val)
random.shuffle(test)

# Select defined number of train, validation and test files
train = tr_val[:args.tr_size]
val = tr_val[args.tr_size:args.tr_size+args.val_size]
ts = test[:args.ts_size]

#########Initialize dataset and dataloader#######

# Create a custom implementation of Pytorch Dataset class
class ContrDataset(Dataset):
    
    def __init__(self, files, max_docs, d_type, n_samples):
        
        self.files = files
        self.unused = files.copy()
        self.ind = 0
        self.max_docs = max_docs
        self.d_type = d_type
        self.n_samples = n_samples
        self.data, self.label = self.get_data()
        
    def get_padding(self, data):
        
        # Adds padding of zeros to a timepoint if it has less documents than the maximum number in the data
        pad = self.max_docs - data.shape[0]
        data = F.pad(input=data, pad=(0, 0, 0, pad), mode='constant', value=0)
        
        return data
    
    def get_data(self):
        
        # Selects dataset randomly and then removes it from the file list
        dataset = random.choice(self.unused)
        self.unused.remove(dataset)
        
        # Loads a new dataset defined by the random index
        a_file = open(args.data_path + self.d_type + "/triplets/" + dataset, "rb")
        d_dict = pickle.load(a_file)
        a_file.close()
        
        data = d_dict['data']
        pair = []
        label = 1
        
        # Chooses randomly positive or negative pair
        positive_pair = random.randint(0,1) 
        
        if positive_pair:
            pair = [data, d_dict['pos_pair']]
            label = 0
        
        else:
            pair = [data, d_dict['neg_pair']]
        
        # Keeps track of the number of timepoints already loaded
        self.ind += args.timepoints*2
        
        return pair, label

    # Function that allows reseting the parameters between epochs
    def reset(self):
        
        self.ind = 0
        self.unused = self.files.copy()

    def __getitem__(self, index):
        
        # Loads new dataset pair when the old one has been used
        if index >= self.ind:
            self.data, self.label = self.get_data()
        
        # Modifies the index based on the amount of data loaded
        timepoint_index = int(str(index)[-2:])
        
        # Selects a timepoint for output
        if index < self.ind - 100:
            timepoint = self.data[0][:,timepoint_index,:]
        else:
            timepoint = self.data[1][:,timepoint_index,:]
        
        # Adds padding if needed
        if timepoint.shape[0] < self.max_docs:
            timepoint = self.get_padding(timepoint)
            
        return timepoint, self.label

    def __len__(self):
        
        # Returns the total number of timepoints in the data
        n_timepoints = self.n_samples*args.timepoints*2
    
        return n_timepoints


# Creates train, validation and test datasets
train_data = ContrDataset(train, max_docs, 'train', len(train))
validation_data = ContrDataset(val, max_docs, 'train', len(val))
test_data = ContrDataset(ts, max_docs, 'test', len(ts))

# Creates dataloaders for train, validation and test data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Print the batch shape
train_features, train_labels = next(iter(train_dataloader))
print(f"Train batch shape: {train_features.size()}")


##############Define the network modules############

# Define the TimepointEmbedding network class
class TimepointEmbedding(nn.Module):
    def __init__(self, fc_input_size, fc_output_size, features, heads, layers, dropout):
        super(TimepointEmbedding, self).__init__()
        
        # define network layers   
        encoder_layer = nn.TransformerEncoderLayer(d_model=features, nhead=heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers) 
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_size, 200*50)
        self.fc2 = nn.Linear(200*50, 200*20)
        self.fc3 = nn.Linear(200*20, 50*10)
        self.fc4 = nn.Linear(50*10, fc_output_size)

    def forward(self, x):
        # define forward pass
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x

# Define the RNN network class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, timepoints):
        super(RNN, self).__init__()

        # define network layers    
        self.blstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=args.lstm_dropout, bidirectional=True, batch_first=True)  
        self.fc = nn.Linear(output_size, int(output_size/2))
        self.fc2 = nn.Linear(int(output_size/2), timepoints)
        
    # Define forward pass
    def forward(self, x):
        # the first value returned by LSTM is all of the hidden states throughout the sequence
        x, _ = self.blstm(x)
        x = self.fc(x.flatten())
        x = self.fc2(x)
        
        return x

#https://medium.com/hackernoon/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
#https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, dist):
        
        if dist == "euclidean":
            euclidean_distance = F.pairwise_distance(output1, output2)
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        elif dist == "cosine":
            self.margin = 0.0
            cosine_similarity = F.cosine_similarity(output1, output2, dim = 0)
            loss_contrastive = torch.mean((1-label) * (1 - cosine_similarity) +
                                          (label) * torch.clamp(cosine_similarity - self.margin, min=0.0))

        return loss_contrastive

# Define input and output size for the fully connected layers
fc_input_size = max_docs*args.emb_size
fc_output_size = args.timepoints

# Define the output size for LSTM
lstm_output_size = 2*args.timepoints*args.hidden_size

# Instantiate the models
tp_emb = TimepointEmbedding(fc_input_size, fc_output_size, args.emb_size, args.n_heads, args.attn_layers, args.attn_dropout).to(device)
rnn = RNN(fc_output_size, args.hidden_size, lstm_output_size, args.lstm_layers, fc_output_size).to(device)

# Print model architectures
#print('TimepointEmbedding: ', tp_emb)
#print('RNN: ', rnn)

# Define optimizer and loss function
optimizer = torch.optim.SGD(list(tp_emb.parameters()) + list(rnn.parameters()), lr=args.lr)
criterion = ContrastiveLoss()

# Keeps count of epochs when training is continued with saved model
epoch_count = 0

## define model and optimizer
if args.continue_tr == 'yes':
    print('Loading model from {}'.format(args.model_path))
    checkpoint = torch.load(args.model_path)
    tp_emb.load_state_dict(checkpoint['tp_emb_state_dict'])
    rnn.load_state_dict(checkpoint['rnn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_count += epoch
    loss = checkpoint['tr_loss']
    print('Model training continues from epoch ', epoch)
    print('Training loss from epoch {}: {}'.format(epoch, loss))

#######Define the train, validation and test iteration functions#######

# Function for training the model
def train_model():
    
    # Set the networks to train mode
    tp_emb.train()
    rnn.train()
    
    losses = []
    
    # Loop over train data batches
    for (x_train, y_train) in train_dataloader:

        # Calculate output
        tp_emb_output = tp_emb(x_train)
        sample1 = tp_emb_output[:args.timepoints]
        sample2 = tp_emb_output[args.timepoints:]
        output1 = rnn(sample1.reshape((1,) + sample1.shape))
        output2 = rnn(sample2.reshape((1,) + sample1.shape))
        
        # Calculate contrastive loss
        contr_loss = criterion(output1, output2, train_data.label, "cosine")
        
        losses.append(contr_loss.item())

        # Backpropagate loss
        optimizer.zero_grad()
        contr_loss.backward()
        optimizer.step()

    return losses

# Function for evaluating the model
def eval_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    tp_emb.eval()
    rnn.eval()
    
    losses = []
    
    # Gradients are not calculated during evaluation
    with torch.no_grad():
        
        # Loop over validation data batches
        for (x_val, y_val) in validation_dataloader:
            
            # Calculate output
            tp_emb_output = tp_emb(x_val)
            sample1 = tp_emb_output[:args.timepoints]
            sample2 = tp_emb_output[args.timepoints:]
            output1 = rnn(sample1.reshape((1,) + sample1.shape))
            output2 = rnn(sample2.reshape((1,) + sample1.shape))
        
            # Calculate contrastive loss
            contr_loss = criterion(output1, output2, validation_data.label, "cosine")
            
            losses.append(contr_loss.item())

    return losses

# Function for testing the model
def test_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    tp_emb.eval()
    rnn.eval()
    
    losses = []
    
    # Gradients are not calculated during testing
    with torch.no_grad():
        
        # Loop over test data batches
        for (x_test, y_test) in test_dataloader:
            
            # Calculate output
            tp_emb_output = tp_emb(x_test)
            sample1 = tp_emb_output[:args.timepoints]
            sample2 = tp_emb_output[args.timepoints:]
            output1 = rnn(sample1.reshape((1,) + sample1.shape))
            output2 = rnn(sample2.reshape((1,) + sample1.shape))
        
            # Calculate contrastive loss
            contr_loss = criterion(output1, output2, test_data.label, "cosine")
            
            losses.append(contr_loss.item())
        
    return losses


#########Train the model##########

# Save loss and accuracy for each epoch
tr_losses = []
val_losses = []

start_time = time.time()

# Calculate training and evaluation loss and accuracy for each epoch
for epoch in range(epoch_count+1, args.epochs+1): 
    tr_loss = train_model()
    val_loss = eval_model()
    tr_losses.append(np.mean(tr_loss))
    val_losses.append(val_loss)

    # Reset the dataset paramters so that all samples are available for the next epoch
    train_data.reset()
    validation_data.reset()

    epoch_res = "|Epoch %d | Training loss : %.3f | Validation loss : %.3f"%(epoch, np.mean(tr_loss), np.mean(val_loss))
    
    # Save results of each epoch to a .txt file
    with open(args.res_path + 'epoch%i.txt' % epoch, 'w') as f:
        f.write(epoch_res)

    # Save the model checkpoint
    torch.save({
            'epoch': epoch,
            'tr_loss': tr_loss,
            'tp_emb_state_dict': tp_emb.state_dict(),
            'rnn_state_dict': rnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, args.model_path)

    print(epoch_res)

end_time = time.time()

print("Minutes used for training: ", int((end_time - start_time)/60))

#tp_emb_params = sum(p.numel() for p in tp_emb.parameters() if p.requires_grad)
#rnn_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
#print("Number of trainable parameters in TimepointEmbedding model: ", tp_emb_params)
#print("Number of trainable parameters in RNN model: ", rnn_params)

##########Get the model accuracy for the test set#########

test_loss = test_model()

print("Test loss : %.3f" %np.mean(test_loss))


##########Save the results#################

#results = {'tr_loss': tr_losses, 'val_loss': val_losses}

# Save loss and accuracy
#file_to_write = open(args.res_path + 'results_dict_10.pkl', 'wb')
#pickle.dump(results, file_to_write)