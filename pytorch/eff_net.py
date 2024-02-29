import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import efficientnet_v2_s, efficientnet_v2_l, EfficientNet_V2_S_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


num_epochs = 200
batch_size = 256
lrn_rate = 0.005
seed = 166
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

def groupdata(data):
    num_samples = len(data)
    num_groups = num_samples // 20
    grouped_data = [data[i*20:(i+1)*20].values.T for i in range(num_groups)]

    return np.array(grouped_data, dtype='float32')


def data_load(n):
    trainX_file = f'../Data/20x20/k-fold/train_x_{n}.csv'
    testX_file = f'../Data/20x20/k-fold/test_x_{n}.csv'  
    trainX = pd.read_csv(trainX_file, header=None, skiprows=1, usecols=range(20))
    testX = pd.read_csv(testX_file, header=None, skiprows=1, usecols=range(20))

    trainX = min_max_normalize(trainX)
    testX = min_max_normalize(testX)

    trainX = groupdata(trainX)
    testX = groupdata(testX)

    trainY_file = f'../Data/20x20/k-fold/train_y_{n}.csv'
    testY_file = f'../Data/20x20/k-fold/test_y_{n}.csv' 
    trainY = pd.read_csv(trainY_file, header=None, skiprows=1)
    testY = pd.read_csv(testY_file, header=None, skiprows=1)
    trainY = trainY.values.flatten()
    testY = testY.values.flatten()

    train_set = TensorDataset(torch.FloatTensor(trainX).unsqueeze(1), torch.LongTensor(trainY))
    val_set = TensorDataset(torch.FloatTensor(testX).unsqueeze(1), torch.LongTensor(testY))

    train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size)

    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)

    # print("Train images size:", images.size())  
    # print("Train labels size:", labels.size())  

    # dataiter = iter(val_loader)
    # images, labels = next(dataiter)

    # print("Test images size:", images.size())  
    # print("Test labels size:", labels.size())

    print(f"\nDataset {n} loaded...")

    return train_loader, val_loader

# ---------------------------------------------------------

# model = efficientnet_v2_s()
# first_conv_layer = model.features[0][0]
# model.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels,
#                                  kernel_size=first_conv_layer.kernel_size,
#                                  stride=first_conv_layer.stride,
#                                  padding=first_conv_layer.padding, bias=False)
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, 2) 
# # print(model)

def initialize_model(lrn_rate):
    model = efficientnet_v2_l()

    first_conv_layer = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels,
                                    kernel_size=first_conv_layer.kernel_size,
                                    stride=first_conv_layer.stride,
                                    padding=first_conv_layer.padding, bias=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2) 

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lrn_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lrn_rate)
    model.to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.75, verbose=True)
    return model, criterion, optimizer, scheduler

# ---------------------------------------------------------

setup_seed(seed)

test_acc = []
best_epoch = []


for n in range(0,10):
    best_acc = -1
    best_acc_epoch = 0
    time_start = time.time()

    model, criterion, optimizer, scheduler = initialize_model(lrn_rate)

    print(f"Loss function: {criterion}, "
        f"Optimizer: {optimizer.__class__.__name__}, "
        f"Learn rate: {lrn_rate:.4f}, "
        f"Batch size: {batch_size}, "
        f"Max epochs: {num_epochs}")
    
    training_logs = []
    train_loader, val_loader = data_load(n)

    for epoch in range(num_epochs):
        running_loss = 0.0
        # all_predictions = []
        # all_labels = []

        # for batch_i, (inputs, labels) in enumerate(train_loader):
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()

        #     _, predicted = torch.max(outputs, 1)
        #     all_predictions.extend(predicted.cpu().numpy())
        #     all_labels.extend(labels.cpu().numpy())

        train_loss = []
        train_acc = [] 
        correct = 0  
        total = 0  
        
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # model, criterion, optimizer, scheduler = initialize_model(lrn_rate)
            optimizer.zero_grad()
            output = model(data)
            output = F.softmax(output, dim=1)
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step() 
            a = target.data.cpu().numpy()
            b = output.detach().cpu().numpy().argmax(1)
            train_acc.append(accuracy_score(a, b))

        # # ---------------------------------------------------------
        val_loss = []
        val_acc = []
        val_prec = []
        val_recall = []
        val_F1 = []
        val_mcc = []
        val_mae = []

        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = F.softmax(output, dim=1)
                loss = criterion(output, target)
                val_loss.append(loss.item()) 

                a = target.data.cpu().numpy()
                b = output.detach().cpu().numpy().argmax(1)

                val_acc.append(accuracy_score(a, b))
                val_prec.append(precision_score(a, b))
                val_recall.append(recall_score(a, b))
                val_F1.append(f1_score(a, b))
                val_mcc.append(matthews_corrcoef(a, b))
                val_mae.append(mean_absolute_error(a, b))

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

        if best_acc < valid_acc:
            best_acc = valid_acc
            best_acc_epoch = epoch
            # print(f'\nEpoch {epoch}, train loss: {np.mean(train_loss):.6f}, valid loss: {np.mean(val_loss):.6f}, train acc: {np.mean(train_acc):.6f}, valid acc: {np.mean(val_acc):.6f}')
            # print("\nSaving best model..")
            model.eval()
            path = f'Model_effs_{n}.pt'
            torch.save(model.state_dict(), path)

            # model = efficientnet_v2_s()
            path_whole = f'Model_effs_{n}w.pt'
            torch.save(model, path_whole) 

    time_end = time.time()
    elapsed_time = time_end - time_start
    print(f"\nTrain cost time = {elapsed_time: .6f}", )

    test_acc.append(best_acc)
    best_epoch.append(best_acc_epoch)
    print(f"\nDataset {n} - Accuracy: {best_acc:.6f}, Best Epoch: {best_acc_epoch}")
   
print('--' * 30)

print("\n") 
for n, (acc, epoch) in enumerate(zip(test_acc, best_epoch)):
    print(f"Dataset {n} - Accuracy: {acc:.8f}, Best Epoch: {epoch}")
    
print("Avg 10-fold CV accuracy:", sum(test_acc) / len(test_acc))


