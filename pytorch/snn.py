import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.cuda import amp
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
# -----------------------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------------------------------------------------------------------------------------------------------------
# Hyper-parameters
num_classes = 2
time_window = 100  # simulating time-steps 
batch_size = 100  # batch size 
epochs = 100  # number of total epochs to run 
num_workers = 4  # number of data loading workers (default: 4) 

amp = False  # automatic mixed precision training (set True if you want to use it) 
optimizer_choice = 'adam'  # use which optimizer. 'sgd' or 'adam' 
momentum = 0.9  # momentum for SGD 
learning_rate = 1e-3  # learning rate 
tau = 2.0  # parameter tau of LIF neuron 

# data_dir = 'path/to/your/mnist/dataset'  # root dir of MNIST dataset 
# out_dir = './logs'  # root dir for saving logs and checkpoint 
# resume = 'path/to/your/checkpoint'  # resume from the checkpoint path (you would specify the actual path here) 

# -----------------------------------------------------------------------------------------------------------------------------------

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# # Max-min normalization
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
#         feats = self.x_data[idx]  # idx row, all 20 cols, with an extra dimension
#         sex = self.y_data[idx]    # idx row, the only col, with an extra dimension
#         return feats, sex

# Raw data
class MyDataset(Dataset):
    def __init__(self, src_file):
        all_data = np.loadtxt(src_file, usecols=range(0, 21), delimiter=",", comments="#", dtype=np.float32, skiprows=1)
       
        self.x_data = torch.tensor(all_data[:, 0:20], dtype=torch.float32).unsqueeze(1).unsqueeze(1).to(device)
        
        self.y_data = torch.tensor(all_data[:, 20], dtype=torch.float32).to(device)  # float32 required

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        feats = self.x_data[idx]  
        sex = self.y_data[idx]    
        return feats, sex

class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(1 * 20, 2, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

# -----------------------------------------------------------------------------------------------------------------------------------

def data_load(n):
    train_file = f'../Data/k-fold/train_{n}.csv'
    test_file = f'../Data/k-fold/test_{n}.csv'   

    train_ds = MyDataset(train_file)  
    test_ds = MyDataset(test_file)    

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"\nDataset {n} loaded...")
    print('batch_size: %.2f, num_epochs: %.2f, tau: %.2f, momentum: %.2f, time_window: %.2f' 
      %(batch_size, epochs, tau, momentum, time_window), '\n')

    return train_loader, test_loader

'''
:return: None

* :ref:`API in English <lif_fc_mnist.main-en>`

.. _lif_fc_mnist.main-en:

The network with FC-LIF structure for classifying MNIST.\n
This function initials the network, starts trainingand shows accuracy on test dataset.
'''

for n in range(0,9):
    start_time = time.time()
    train_loader, test_loader = data_load(n)

    setup_seed(166)
    net = SNN(tau=tau)
    print(net)
    net.to(device)

    scaler = None
    if amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    acc_recd = []
    prec_recd = []
    recall_recd = []
    F1_recd = []
    MCC_recd = []
    MAE_recd = []

    optimizer = None
    if optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(optimizer_choice)

    # if resume:
    #     checkpoint = torch.load(resume, map_location='cpu')
    #     net.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     max_test_acc = checkpoint['max_test_acc']
    
    # out_dir = os.path.join(out_dir, f'Time{time_window}_batch{batch_size}_{optimizer_choice}_lr{learning_rate}')

    # if amp:
    #     out_dir += '_amp'

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #     print(f'Mkdir {out_dir}.')

    # with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    #     args_txt.write(str(args))

    # writer = SummaryWriter(out_dir, purge_step=start_epoch)
    # with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    #     args_txt.write(str(args))
    #     args_txt.write('\n')
    #     args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, epochs):
        
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device).long()
            label_onehot = F.one_hot(label, num_classes).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(time_window):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / time_window
                    loss = F.mse_loss(out_fr, label_onehot)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(time_window):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / time_window
                loss = F.mse_loss(out_fr, label_onehot)
                
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        print(f'epoch {epoch} train_loss = {train_loss: .6f}, train_acc = {train_acc: .6f}')

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        all_targets = []
        all_predictions = []
        all_scores = []

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device).long()
                label_onehot = F.one_hot(label, num_classes).float()
                out_fr = 0.
                for t in range(time_window):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / time_window
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                
                predicted = out_fr.argmax(1)
                all_targets.extend(label.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_scores.extend(out_fr.cpu().numpy())

                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

                
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        print(f'epoch {epoch} test_loss = {test_loss: .6f}, test_acc = {test_acc: .6f}')

        precision = precision_score(all_targets, all_predictions, average='macro')
        recall = recall_score(all_targets, all_predictions, average='macro')
        f1 = f1_score(all_targets, all_predictions, average='macro')
        mcc = matthews_corrcoef(all_targets, all_predictions)

        mae = mean_absolute_error(np.eye(num_classes)[all_targets], all_scores)

        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)

        # save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_acc_epoch = epoch
            save_max = True

        # checkpoint = {
        #     'net': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'epoch': epoch,
        #     'max_test_acc': max_test_acc
        # }

        # if save_max:
        #     torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        # torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        acc_recd.append(test_acc)
        prec_recd.append(precision)
        recall_recd.append(recall)
        F1_recd.append(f1)
        MCC_recd.append(mcc)
        MAE_recd.append(mae)


        print(f'epoch {epoch} train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
        print(f'epoch {epoch} test_loss = {test_loss: .4f}, test_acc = {test_acc: .4f}, precision = {precision: .4f}, recall = {recall: .4f}, F1 = {f1: .4f}, MCC = {mcc: .4f}, MAE = {mae: .4f}')
        print(f'train speed ={train_speed: .4f} entires/s, test speed ={test_speed: .4f} entires/s\n')
        # print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    print(f'max_test_acc ={max_test_acc: .4f} at epoch {max_acc_epoch}')
    print('--------------------------------------------------')



    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The operation took {elapsed_time} seconds.")

    # save model
    print("\nSaving trained model state_dict ")
    net.eval()
    path = f"F:/programming/SNN_HP/Model_basic_snn_{n}.pt"
    torch.save(net.state_dict(), path)

    model = SNN(tau=tau)
    path_whole1 = f"F:/programming/SNN_HP/Model_basic_snn_{n}w.pt"
    path_whole2 = f"F:/programming/SNN_HP/Model_basic_snn_{n}w.pth"
    torch.save(model, path_whole1)
    torch.save(model, path_whole2)
    

    print(f'Avg. test_acc = {np.mean(acc_recd): .4f}, precision = {np.mean(prec_recd): .4f}, recall = {np.mean(recall_recd): .4f}, F1 = {np.mean(F1_recd): .4f}, MCC = {np.mean(MCC_recd): .4f}, MAE = {np.mean(MAE_recd): .4f}')


    # # plot
    # net.eval()

    # output_layer = net.layer[-1] 
    # output_layer.v_seq = []
    # output_layer.s_seq = []
    # def save_hook(m, x, y):
    #     m.v_seq.append(m.v.unsqueeze(0))
    #     m.s_seq.append(y.unsqueeze(0))

    # output_layer.register_forward_hook(save_hook)


    # with torch.no_grad():
    #     img, label = test_dataset[0]
    #     img = img.to(device)
    #     out_fr = 0.
    #     for t in range(time_window):
    #         encoded_img = encoder(img)
    #         out_fr += net(encoded_img)
    #     out_spikes_counter_frequency = (out_fr / time_window).cpu().numpy()
    #     print(f'Firing rate: {out_spikes_counter_frequency}')

    #     output_layer.v_seq = torch.cat(output_layer.v_seq)
    #     output_layer.s_seq = torch.cat(output_layer.s_seq)
    #     v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  
    #     np.save("v_t_array.npy",v_t_array)
    #     s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  
    #     np.save("s_t_array.npy",s_t_array)

