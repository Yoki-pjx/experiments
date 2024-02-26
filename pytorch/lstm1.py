import numpy as np
import pandas as pd
import time
import torch
import random
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

seed = 166

in_dim = 20
hidden_dim = 16
num_layers = 1
bidirectional = True
num_classes = 2

epochs = 2000
batch_size = 256
lrn_rate = 0.005
sequence_length = 16

patience = 50
verbose = True

if num_layers > 1:
    dropout = 0.1
else:
    dropout = 0


# ---------------------------------------------------------

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------

def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min())

def groupdata(data, sequence_length):
    num_samples = len(data)
    num_groups = num_samples // sequence_length
    grouped_data = [data[i*sequence_length:(i+1)*sequence_length].values.T for i in range(num_groups)]

    return np.array(grouped_data, dtype='float32')

def data_load(n):
    trainX_file = f'../Data/time_series/k-fold/whole_block/train_x_{n}.csv'
    testX_file = f'../Data/time_series/k-fold/whole_block/test_x_{n}.csv'  
    trainX = pd.read_csv(trainX_file, header=None, skiprows=1, usecols=range(20))
    testX = pd.read_csv(testX_file, header=None, skiprows=1, usecols=range(20))

    trainX = min_max_normalize(trainX)
    testX = min_max_normalize(testX)

    trainX = groupdata(trainX, sequence_length)
    testX = groupdata(testX, sequence_length)

    trainY_file = f'../Data/time_series/k-fold/whole_block/train_y_{n}.csv'
    testY_file = f'../Data/time_series/k-fold/whole_block/test_y_{n}.csv'
    trainY = pd.read_csv(trainY_file, header=None, skiprows=1)
    testY = pd.read_csv(testY_file, header=None, skiprows=1)
    trainY = trainY.values.flatten()
    testY = testY.values.flatten()

    train_set = torch.utils.data.TensorDataset(torch.FloatTensor(trainX), torch.LongTensor(trainY))
    val_set = torch.utils.data.TensorDataset(torch.FloatTensor(testX), torch.LongTensor(testY))

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)

    print(f"\nDataset {n} loaded...")

    return train_loader, val_loader


# ---------------------------------------------------------

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

    # With GRU
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
        
    # Without GRU
    def forward(self, x):

        x = x.permute(2, 0, 1)
        lstm_out, _ = self.lstm(x)
        avg_pool_l = torch.mean(lstm_out.permute(1, 0, 2), 1)
        max_pool_l, _ = torch.max(lstm_out.permute(1, 0, 2), 1)
        
        x = torch.cat((avg_pool_l, max_pool_l), 1)
        y = self.fc(x)
        
        return y

# --------------------------------------------------------- 
    
def initialize_model(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size, lrn_rate):
    model = lstm(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rate)
    model.to(device)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    return model, loss_func, optimizer, scheduler

# ---------------------------------------------------------
setup_seed(seed)

test_acc = []
best_epoch = []

for n in range(0,10):
    # valid_loss_min = np.Inf
    # patience = patience
    # # current number of epochs, where validation loss didn't increase
    # p = 0
    # # whether training should be stopped
    # stop = False
    best_acc = -1
    best_acc_epoch = 0
    time_start = time.time()
    
    model, loss_func, optimizer, scheduler = initialize_model(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size, lrn_rate)

    print(f"Loss function: {loss_func}, "
            f"Optimizer: {optimizer.__class__.__name__}, "
            f"Learn rate: {lrn_rate:.4f}, "
            f"Batch size: {batch_size}, "
            f"Max epochs: {epochs}")

    training_logs = []
    train_loader, val_loader = data_load(n)

    print("\nStarting training")
    for epoch in range(epochs):
        # print(time.ctime(), 'Epoch:', e)
        train_loss = []
        train_acc = []
        model, loss_func, optimizer, scheduler = initialize_model(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size, lrn_rate)
        # train model
        model.train()
        for batch_i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            train_loss.append(loss.item())

            a = target.data.cpu().numpy()
            b = output.detach().cpu().numpy().argmax(1)
            train_acc.append(accuracy_score(a, b))

            loss.backward()
            optimizer.step()

    # ---------------------------------------------------------
        # evaluate model    
        val_loss = []
        val_acc = []
        val_prec = []
        val_recall = []
        val_F1 = []
        val_mcc = []
        val_mae = []
    
        model.eval()
        with torch.no_grad():
            for batch_i, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = loss_func(output, target)
                val_loss.append(loss.item()) 

                a = target.data.cpu().numpy()
                b = output.detach().cpu().numpy().argmax(1)

                val_acc.append(accuracy_score(a, b))
                val_prec.append(precision_score(a, b))
                val_recall.append(recall_score(a, b))
                val_F1.append(f1_score(a, b))
                val_mcc.append(matthews_corrcoef(a, b))
                val_mae.append(mean_absolute_error(a, b))

        if epoch % 25 == 0 and verbose:
            print(f'\nEpoch {epoch}, train loss: {np.mean(train_loss):.6f}, valid loss: {np.mean(val_loss):.6f}, train acc: {np.mean(train_acc):.6f}, valid acc: {np.mean(val_acc):.6f}')
            print("Metrics for test data:  "
                f"accuracy = {np.mean(val_acc):.4f}, "
                f"precision = {np.mean(val_prec):.4f}, "
                f"recall = {np.mean(val_recall):.4f}, "
                f"F1 = {np.mean(val_F1):.4f}, "
                f"mcc = {np.mean(val_mcc):.4f}, "
                f"mae = {np.mean(val_mae):.4f}")

        training_logs.append([epoch, np.mean(train_loss), np.mean(val_loss), np.mean(train_acc), np.mean(val_acc)])

        scheduler.step(np.mean(val_loss))

        valid_acc = np.mean(val_acc)
        valid_loss = np.mean(val_loss)
        # if valid_loss <= valid_loss_min:
        #     # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        #     # torch.save(model.state_dict(), 'model.pt')
        #     valid_loss_min = valid_loss
        #     p = 0
        # # check if validation loss didn't improve
        # if valid_loss > valid_loss_min:
        #     p += 1
        #     # print(f'{p} epochs of increasing val loss')
        #     if p > patience:
        #         print('Stopping training')
        #         stop = True
        #         break        
        # if stop:
        #     break

        # Save best model
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_acc_epoch = epoch

            print(f'\nEpoch {epoch}, train loss: {np.mean(train_loss):.6f}, valid loss: {np.mean(val_loss):.6f}, train acc: {np.mean(train_acc):.6f}, valid acc: {np.mean(val_acc):.6f}')
            print("Metrics for test data:  "
                f"accuracy = {np.mean(val_acc):.4f}, "
                f"precision = {np.mean(val_prec):.4f}, "
                f"recall = {np.mean(val_recall):.4f}, "
                f"F1 = {np.mean(val_F1):.4f}, "
                f"mcc = {np.mean(val_mcc):.4f}, "
                f"mae = {np.mean(val_mae):.4f}")
            # print("\nSaving best model..")
            model.eval()
            path = f'Model_lstm1_{n}.pt'
            torch.save(model.state_dict(), path)

            model = lstm(in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size)
            path_whole = f'Model_lstm1_{n}w.pt'
            torch.save(model, path_whole) 

    time_end = time.time()
    elapsed_time = time_end - time_start
    print(f"Train cost time = {elapsed_time: .6f}", )

    test_acc.append(best_acc)
    best_epoch.append(best_acc_epoch)
    print(f"\nDataset {n} - Accuracy: {best_acc:.6f}, Best Epoch: {best_acc_epoch}")

        # Checkpoint
        # checkpoint = torch.load('model.pt')      
        # model.load_state_dict(checkpoint)
    
    print('--' * 30)



print("\n") 
for n, (acc, epoch) in enumerate(zip(test_acc, best_epoch)):
    print(f"Dataset {n} - Accuracy: {acc:.8f}, Best Epoch: {epoch}")
    
print("Avg 10-fold CV accuracy:", sum(test_acc) / len(test_acc))

        
   