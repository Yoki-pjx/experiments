import numpy as np
import pandas as pd
import time
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split, GroupKFold, GroupShuffleSplit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

in_dim = 20
hidden_dim = 16
num_layers = 1
bidirectional = True
num_classes = 2
batch_size = 256
lrn_rate = 0.005
sequence_length = 16
patience = 50

if num_layers > 1:
    dropout = 0.1
else:
    dropout = 0

n = 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def groupdata(data, sequence_length):
    num_samples = len(data)
    num_groups = num_samples // sequence_length
    grouped_data = [data[i*16:(i+1)*16].values.T for i in range(num_groups)]
    return np.array(grouped_data, dtype='float32')

trainX_file = f'./Data/time_series/k-fold/whole_block/train_x_{n}.csv'
testX_file = f'./Data/time_series/k-fold/whole_block/test_x_{n}.csv'  
trainX = pd.read_csv(trainX_file, header=None, skiprows=1, usecols=range(20))
testX = pd.read_csv(testX_file, header=None, skiprows=1, usecols=range(20))
trainX = groupdata(trainX, sequence_length)
testX = groupdata(testX, sequence_length)

trainY_file = f'./Data/time_series/k-fold/whole_block/train_y_{n}.csv'
testY_file = f'./Data/time_series/k-fold/whole_block/test_y_{n}.csv'
trainY = pd.read_csv(trainY_file, header=None, skiprows=1)
testY = pd.read_csv(testY_file, header=None, skiprows=1)
trainY = trainY.values.flatten()
testY = testY.values.flatten()


class lstm(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(lstm, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout/2, bidirectional=self.bidirectional,
                            batch_first=True)
        # self.gru = nn.GRU(self.hidden_dim * 2, self.hidden_dim, bidirectional=self.bidirectional, batch_first=True)


        self.fc = nn.Sequential(
            nn.Linear(64, int(hidden_dim)),
            nn.SELU(True),
            nn.Dropout(p=dropout),
            nn.Linear(int(hidden_dim), num_classes),
        )

    # def forward(self, x):

    #     x = x.permute(2, 0, 1)
    #     lstm_out, _ = self.lstm(x)
    #     gru_out, _ = self.gru(lstm_out)
    #     avg_pool_l = torch.mean(lstm_out.permute(1, 0, 2), 1)
    #     max_pool_l, _ = torch.max(lstm_out.permute(1, 0, 2), 1)
        
    #     avg_pool_g = torch.mean(gru_out.permute(1, 0, 2), 1)
    #     max_pool_g, _ = torch.max(gru_out.permute(1, 0, 2), 1)

    #     x = torch.cat((avg_pool_g, max_pool_g, avg_pool_l, max_pool_l), 1)
    #     y = self.fc(x)
        
    #     return y
    def forward(self, x):

        x = x.permute(2, 0, 1)
        lstm_out, _ = self.lstm(x)
        avg_pool_l = torch.mean(lstm_out.permute(1, 0, 2), 1)
        max_pool_l, _ = torch.max(lstm_out.permute(1, 0, 2), 1)
        
        x = torch.cat((avg_pool_l, max_pool_l), 1)
        y = self.fc(x)
        
        return y

def train_net(train_loader, val_loader, patience, model, loss_func, optimizer, scheduler, verbose):
    valid_loss_min = np.Inf
    patience = patience
    # current number of epochs, where validation loss didn't increase
    p = 0
    # whether training should be stopped
    stop = False

    epochs = 5
    training_logs = []

    for epoch in range(epochs):
        # print(time.ctime(), 'Epoch:', e)
        train_loss = []
        train_acc = []
        model.train()
        for batch_i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            train_loss.append(loss.item())

            a = target.data.cpu().numpy()
            b = output.detach().cpu().numpy().argmax(1)
            train_acc.append(accuracy_score(a, b))

            loss.backward()
            optimizer.step()

        val_loss = []
        val_acc = []
        for batch_i, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)

            loss = loss_func(output, target)
            val_loss.append(loss.item()) 
            a = target.data.cpu().numpy()
            b = output.detach().cpu().numpy().argmax(1)
            val_acc.append(accuracy_score(a, b))

        if epoch % 100 == 0 and verbose:
            print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train acc: {np.mean(train_acc):.4f}, valid acc: {np.mean(val_acc):.4f}')

        training_logs.append([epoch, np.mean(train_loss), np.mean(val_loss), np.mean(train_acc), np.mean(val_acc)])

        scheduler.step(np.mean(val_loss))

        valid_loss = np.mean(val_loss)
        if valid_loss <= valid_loss_min:
            # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
            p = 0

        # check if validation loss didn't improve
        if valid_loss > valid_loss_min:
            p += 1
            # print(f'{p} epochs of increasing val loss')
            if p > patience:
                print('Stopping training')
                stop = True
                break        

        if stop:
            break

    checkpoint = torch.load('model.pt')      
    model.load_state_dict(checkpoint)
        
    return model

def initialize_model(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size, lrn_rate):
    torch.manual_seed(42)
    model = lstm(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rate)
    model.cuda()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, verbose=True)
    return model, loss_func, optimizer, scheduler



def train_net_folds(X, X_test, y, folds, batch_size, patience, verbose):

    oof = np.zeros((len(X), 2))
    prediction = np.zeros((len(X_test), 2))
    scores = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        train_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid), torch.LongTensor(y_valid))

        train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)
        
        model, loss_func, optimizer, scheduler = initialize_model(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size, lrn_rate)
        model = train_net(train_loader, val_loader, patience, model, loss_func, optimizer, scheduler, verbose)
        
        y_pred_valid = []
        for batch_i, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            p = model(data)
            pred = p.cpu().detach().numpy()
            y_pred_valid.extend(pred)
            
        y_pred = []
        for i, data in enumerate(testX):
            p = model(torch.FloatTensor(data).unsqueeze(0).cuda())
            y_pred.append(p.cpu().detach().numpy().flatten())
            
        oof[valid_index] = np.array(y_pred_valid)
        scores.append(accuracy_score(y_valid, np.array(y_pred_valid).argmax(1)))

        prediction += y_pred

    prediction /= n_fold
    
    prediction = np.array(prediction).argmax(1)
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    print('--' * 50)
    
    return oof, prediction

n_fold = 2
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
oof, prediction = train_net_folds(trainX, testX, trainY, folds, batch_size, patience, True)