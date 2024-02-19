# Binary danger classification
import torch
import time
import math
import random
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# 0. get started
print("\nDanger detection using PyTorch ")
seed = 100  

# 1. create Dataset and DataLoader objects
print("\nCreating train and test Datasets ")
batch_size = 512
lrn_rate = 0.005
max_epochs = 500
ep_log_interval = 20


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Load dataset
# class MyDataset(Dataset):
#     def __init__(self, src_file):
#         all_data = np.loadtxt(src_file, usecols=range(0, 21), delimiter=",", comments="#", dtype=np.float32, skiprows=1)

#         # Calculate min and max for each feature (column)
#         min_values = np.min(all_data[:, 0:20], axis=0)
#         max_values = np.max(all_data[:, 0:20], axis=0)
        
#         # Apply max-min normalization for each feature (column)
#         normalized_data = (all_data[:, 0:20] - min_values) / (max_values - min_values)
        
#         self.x_data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1).to(device)
        
#         self.y_data = torch.tensor(all_data[:, 20], dtype=torch.float32).to(device)  # float32 required

#     def __len__(self):
#         return len(self.x_data)

#     def __getitem__(self, idx):
#         feats = self.x_data[idx]  
#         sex = self.y_data[idx]    
#         return feats, sex
    
# Load dataset
class MyDataset(Dataset):
    def __init__(self, src_file):
        all_data = np.loadtxt(src_file, usecols=range(0, 21), delimiter=",", comments="#", dtype=np.float32, skiprows=1)
       
        self.x_data = torch.tensor(all_data[:, 0:20], dtype=torch.float32).to(device)
        
        self.y_data = torch.tensor(all_data[:, 20], dtype=torch.float32).to(device)  # float32 required

        self.y_data = self.y_data.reshape(-1,1)  # 2-D required

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        feats = self.x_data[idx]  
        sex = self.y_data[idx]    
        return feats, sex

# ---------------------------------------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hid1 = torch.nn.Linear(20,10) # 21-(10-10)-1
        self.hid2 = torch.nn.Linear(10,10)
        self.hid3 = torch.nn.Linear(10,10)
        self.oupt = torch.nn.Linear(10,1)

        torch.nn.init.xavier_uniform_(self.hid1.weight) 
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight) 
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.hid3.weight) 
        torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight) 
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self,x):
        z = torch.tanh(self.hid1(x))
        z = torch.tanh(self.hid2(z))
        z = torch.tanh(self.hid3(z))
        z = torch.sigmoid(self.oupt(z))  # for BCELoss()
        return z
        
# ---------------------------------------------------------

def metrics(model, ds, thresh=0.5):
  # note: N = total number of items = TP + FP + TN + FN
  # accuracy  = (TP + TN)  / N
  # precision = TP / (TP + FP)
  # recall    = TP / (TP + FN)
  # F1        = 2 / [(1 / precision) + (1 / recall)]
  # mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

  tp = 0; tn = 0; fp = 0; fn = 0
  for i in range(len(ds)):
    inpts = ds[i][0]         # Tuple style
    target = ds[i][1]        # float32  [0.0] or [1.0]
    with torch.no_grad():
      p = model(inpts)       # between 0.0 and 1.0

    # should really avoid 'target == 1.0'
    if target > 0.5 and p >= thresh:    # TP
      tp += 1
    elif target > 0.5 and p < thresh:   # FP
      fp += 1
    elif target < 0.5 and p < thresh:   # TN
      tn += 1
    elif target < 0.5 and p >= thresh:  # FN
      fn += 1

  N = tp + fp + tn + fn
  if N != len(ds):
    print("FATAL LOGIC ERROR in metrics()")

  accuracy = (tp + tn) / (N * 1.0)
  precision = (1.0 * tp) / (tp + fp)
  recall = (1.0 * tp) / (tp + fn)
  f1 = 2.0 / ((1.0 / precision) + (1.0 / recall))
  mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  return (accuracy, precision, recall, f1, mcc)  # as a Tuple

# ---------------------------------------------------------

def data(n):
    train_file = f'../Data/k-fold/train_{n}.csv'
    test_file = f'../Data/k-fold/test_{n}.csv'  

    train_ds = MyDataset(train_file)  
    test_ds = MyDataset(test_file)    

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"\nDataset {n} loaded...")

    return train_loader, test_loader, train_ds, test_ds

# ---------------------------------------------------------
for n in range(0,9):
  setup_seed(seed)
  time_start = time.time()
  train_loader, test_loader, train_ds, test_ds = data(n)

  # 2. create neural network
  print("\nCreating 20-(10-10-10)-1 binary NN classifier \n")
  net = Net().to(device)
  net.train()  # set training mode

  # 3. train network
  loss_func = torch.nn.BCELoss()  # binary cross entropy
  # loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate)


  print(f"Loss function: {loss_func}, "
        f"Optimizer: {optimizer.__class__.__name__}, "
        f"Learn rate: {lrn_rate:0.3f}, "
        f"Batch size: {batch_size}, "
        f"Max epochs: {max_epochs}")


  print("\nStarting training")
  for epoch in range(0, max_epochs):
    # epoch_loss = 0.0            # for one full epoch
#     for (batch_idx, batch) in enumerate(train_loader):
#       X = batch[0]             
#       Y = batch[1]             
#       oupt = net(X)            

#       loss_val = loss_func(oupt, Y)   # a tensor
#       epoch_loss += loss_val.item()  # accumulate
#       optimizer.zero_grad() # reset all gradients
#       loss_val.backward()   # compute new gradients
#       optimizer.step()      # update all weights

#     if epoch % ep_log_interval == 0:
#       print("epoch = %4d   loss = %8.4f" % \
#         (epoch, epoch_loss))
#   print("Train completed... ")

    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()  # 累加每个批次的损失

    # 计算平均训练损失
    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # ---------------------------------------------------------

    # 4. evaluate model
    net.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = loss_func(output, target)
            total_test_loss += loss.item()
    # 计算平均训练损失
    avg_test_loss = total_test_loss / len(test_loader.dataset)

    print(f'Epoch: {epoch} - Average Training Loss: {avg_train_loss:.6f}, Average Test Loss: {avg_test_loss:.6f}')

    metrics_train = metrics(net, train_ds, thresh=0.5)
    print("Metrics for train data: "
        f"accuracy = {metrics_train[0]:0.4f}, "
        f"precision = {metrics_train[1]:0.4f}, "
        f"recall = {metrics_train[2]:0.4f}, "
        f"F1 = {metrics_train[3]:0.4f}, "
        f"mcc = {metrics_train[4]:0.4f}")

  metrics_test = metrics(net, test_ds, thresh=0.5)
  print("\nMetrics for test data: "
        f"accuracy = {metrics_test[0]:0.4f}, "
        f"precision = {metrics_test[1]:0.4f}, "
        f"recall = {metrics_test[2]:0.4f}, "
        f"F1 = {metrics_test[3]:0.4f}, "
        f"mcc = {metrics_test[4]:0.4f}")
  
  time_end = time.time()
  print("Train cost time = ", time_end - time_start)

  # 5. save model
  print("\nSaving trained model state_dict ")
  net.eval()
  path = f'Model_nn_{n}.pt'
  torch.save(net.state_dict(), path)


print("Binary classification end... ")

