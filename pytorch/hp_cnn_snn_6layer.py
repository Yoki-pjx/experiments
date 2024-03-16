# Libraries
import random
import numpy as np
import torch,time
import torch.nn as nn
import torch.nn.functional as Functional
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import math
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------------------------------------------------------------------------------------------------------------
# Hyper-parameters

# num_updates = 1 # meta-parameter update epochs, not used in this demo 
thresh = 0.45 # threshold 
lens = 0.5 # hyper-parameter in the approximate firing functions 
decay = 0.75  # the decay constant of membrane potentials 
w_decay = 0.9 # weight decay factor 
num_classes = 2
batch_size = 100
tau_w = 40 # synaptic filtering constant 
num_epochs = 3
learning_rate = 5e-4
# lp_learning_rate = 5e-4  # learning rate of meta-local parameters 
# gp_learning_rate = 1e-3 # learning rate of gp-based parameters 
time_window = 8 # time windows
seed = 166
# -----------------------------------------------------------------------------------------------------------------------------------
# Model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ActFun(torch.autograd.Function):  
    '''
    Approaximation function of spike firing rate function
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()
 
cfg_cnn = [
    (1, 64, 1, 1, 3),

    (64, 128, 1, 1, 3),

    (128, 128, 1, 1, 3),

]

cnn_dim = [20, 10, 5, 2]
fc_dim = cfg_cnn[-1][1] * cnn_dim[-1] * cnn_dim[-1]

cfg_fc = [256, 2]

probs = 0.1 # dropout rate
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, lr_decay_epoch=25): 
    """
    Decay learning rate by a factor of 0.8 every lr_decay_epoch epochs.
    """
    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8
    return optimizer




class SNN_Model(nn.Module): 

    def __init__(self, p=0.5):
        super(SNN_Model, self).__init__()

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(fc_dim, cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )

        self.fc2.weight.data = self.fc2.weight.data * 0.5
        self.mask1 = p > torch.rand(self.fc1.weight.size(), device=device)
        self.mask2 = p > torch.rand(self.fc2.weight.size(), device=device)
        self.mask1 = self.mask1.float()
        self.mask2 = self.mask2.float()


        self.alpha1 = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((.0 * torch.zeros(1,cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((.0 * torch.zeros(1,cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((1e-2 * torch.ones(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((1e-2 * torch.ones(1)).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.ones(1, fc_dim)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.ones(1, cfg_fc[0])).cuda(), requires_grad=True)

    def mask_weight(self): 
        self.fc1.weight.data = self.fc1.weight.data * self.mask1
        self.fc2.weight.data = self.fc2.weight.data * self.mask2

    def produce_hebb(self): 
        hebb1 = torch.zeros(fc_dim, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

    def forward(self, input, hebb1, hebb2, wins = time_window):

        batch_size = input.size(0)

        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cnn_dim[0], cnn_dim[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cnn_dim[1], cnn_dim[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cnn_dim[2], cnn_dim[2], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(wins):

            decay_factor = np.exp(- step / tau_w)

            x = input

            c1_mem, c1_spike = mem_update_nonplastic(self.conv1, x.float(), c1_spike, c1_mem)

            x = Functional.avg_pool2d(c1_spike, 2)   

            c2_mem, c2_spike = mem_update_nonplastic(self.conv2, Functional.dropout(x * decay_factor, p=probs, training=self.training), c2_spike,
                                               c2_mem)
            x = Functional.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update_nonplastic(self.conv3, Functional.dropout(x * decay_factor, p=probs, training=self.training), c3_spike,
                                               c3_mem)

            x = Functional.avg_pool2d(c3_spike, 2)   

            x = x.view(batch_size, -1).float()                  

            h1_mem, h1_spike, hebb1 = mem_update(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1, x * decay_factor, h1_spike,
                                                 h1_mem, hebb1)
            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike = mem_update_nonplastic(self.fc2, h1_spike * decay_factor, h2_spike, h2_mem)

            buf = h2_mem.gt(thresh).float().max(dim=1)[0].detach_().mean()
            if buf > 0.9 and step > 0:
                break

        outs = (h2_mem / thresh).clamp(max=1.1)

        return outs, h1_mem, hebb1.data, hebb2.data, self.eta1, self.eta2


def mem_update(fc, alpha, beta, gamma, eta, inputs,  spike, mem, hebb):
    '''
    Update the membrane potentials
    Note that : The only difference between the GP and HP model is whether to use hebb-based local variables
    :param fc: linear opetrations 
    :param alpha: the weight of hebb module 
    :param beta: the meta-local parameters to control the learning rate 
    :param gamma: the meta-local parameters to control the weight decay
    :param eta: the meta-local parameters  of sliding threshold
    :return: current membrane potentials, spikes, and local states
    '''
    state = fc(inputs) + alpha * inputs.mm(hebb)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)
    # Update local modules
    hebb = w_decay * hebb - torch.bmm((inputs * beta.clamp(min=0.)).unsqueeze(2),
                                   ((mem / thresh) - eta).unsqueeze(1)).mean(dim=0).squeeze()
    hebb = hebb.clamp(min=-4, max=4)
    return mem, now_spike.float(), hebb

def mem_update_nonplastic(fc, inputs, spike, mem):
    state = fc(inputs)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)
    return mem, now_spike.float() 


# -----------------------------------------------------------------------------------------------------------------------------------
# Main

# Load dataset
# Max-min normalization
class MyDataset(Dataset):
    def __init__(self, src_file):
        all_data = np.loadtxt(src_file, usecols=range(0, 21), delimiter=",", comments="#", dtype=np.float32, skiprows=1)

        # Calculate min and max for each feature (column)
        min_values = np.min(all_data[:, 0:20], axis=0)
        max_values = np.max(all_data[:, 0:20], axis=0)
        
        # Apply max-min normalization for each feature (column)
        normalized_data = (all_data[:, 0:20] - min_values) / (max_values - min_values)
        
        # Reshape the data to combine every 20 rows into one data point
        num_data_points = len(normalized_data) - 19  # Adjust the length to be divisible by 20
        reshaped_data = np.zeros((num_data_points, 20, 20))
        for i in range(num_data_points):
            reshaped_data[i] = normalized_data[i:i+20]

        self.x_data = torch.tensor(reshaped_data, dtype=torch.float32).unsqueeze(1).to(device)
        
        self.y_data = torch.tensor(all_data[19:, 20], dtype=torch.float32).to(device)  # Adjusted to match the reshaped data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        feats = self.x_data[idx]  
        sex = self.y_data[idx]    
        return feats, sex

def data(n):
    train_file = f'../Data/20x20/k-fold/train_x_{n}.csv'
    test_file = f'../Data/20x20/k-fold/test_x_{n}.csv'  

    train_ds = MyDataset(train_file)  
    test_ds = MyDataset(test_file)    

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"\nDataset {n} loaded...")
    print('thresh: %.2f, decay: %.2f, batch_size: %d, num_epochs: %d, tau_w: %d, w_decay: %.1f, time_window: %d, cfg_fc:' 
      %(thresh, decay, batch_size, num_epochs, tau_w, w_decay, time_window), cfg_fc, '\n')

    return train_loader, test_loader


for n in range(9,10):

    train_loader, test_loader = data(n)

    best_acc = 0.  # best test accuracy 
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch 
    acc_record = list([])  
    loss_train_record = list([]) 
    loss_test_record = list([])  

    criterion = nn.MSELoss()  

    total_best_acc = []       
    total_acc_record = []     
    total_hid_state = []      

    exp_num = 1 
    for exp_index in range(exp_num):
        start_time_w = time.time()
        
        setup_seed(seed) 
        snn = SNN_Model()
        snn.to(device)
        optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

        max_acc = -1
        max_acc_epoch = 0

        acc_record = [] 
        hebb1, hebb2 = snn.produce_hebb()
        for epoch in range(num_epochs):
            running_loss = 0.
            snn.train()
            start_time = time.time()

            total = 0.
            correct = 0.

            for i, (images, targets) in enumerate(train_loader):
                snn.zero_grad()
                optimizer.zero_grad()

                images = images.float().to(device)
                outputs, spikes, hebb1, hebb2, eta1, eta2 = snn(input=images,hebb1 = hebb1, hebb2 = hebb2, wins = time_window)
                targets_ = torch.zeros(targets.shape[0], num_classes, device=device).scatter_(1, targets.view(-1, 1).long(), 1).float()
                # targets_ = torch.zeros(batch_size, num_classes).scatter_(1, targets.view(-1, 1), 1)

                loss = criterion(outputs, targets_)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += float(targets.size(0))

                correct += float(predicted.eq(targets).sum().item())

                if i % (len(train_loader)/batch_size/4) == 0:
                    print('Train Accuracy of the model : %.3f' % (100 * correct / total))
                
            loss_train_record.append(running_loss)
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'%(epoch+1, num_epochs, i+1, len(train_loader),running_loss ))


            print('Running time:', time.time() - start_time)
            correct = 0.
            total = 0.
            running_loss = 0.

            tp = 0; tn = 0; fp = 0; fn = 0
            mae = 0.0  

            all_targets = []
            all_predictions = []
            all_scores = []

            optimizer = lr_scheduler(optimizer, epoch)
            snn.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    outputs, sumspike, _, _, _, _ = snn(input=inputs, hebb1=hebb1, hebb2=hebb2, wins=time_window)

                    targets_ = torch.zeros(targets.shape[0], num_classes, device=device).scatter_(1, targets.view(-1, 1).long(), 1).float()                
                    loss = criterion(outputs, targets_)

                    _, predicted = outputs.max(1)
                    total += float(targets.size(0))

                    correct += float(predicted.eq(targets).sum().item())

                    for t, p in zip(targets, predicted):
                        if t > 0.5 and p >= 0.5:  # TP
                            tp += 1
                        elif t > 0.5 and p < 0.5:  # FN
                            fn += 1
                        elif t < 0.5 and p < 0.5:  # TN
                            tn += 1
                        elif t < 0.5 and p >= 0.5:  # FP
                            fp += 1

                    mae += abs(targets - predicted).sum().item()

                N = tp + fp + tn + fn
                accuracy = (tp + tn) / (N * 1.0)
                precision = (1.0 * tp) / (tp + fp) if tp + fp > 0 else 0
                recall = (1.0 * tp) / (tp + fn) if tp + fn > 0 else 0
                f1 = 2.0 / ((1.0 / precision) + (1.0 / recall)) if precision + recall > 0 else 0
                mcc = ((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
                mae /= len(test_loader.dataset)  

                acc = 100. * float(correct) / float(total)

            if acc > max_acc:
                max_acc = acc
                max_acc_epoch = epoch

            print('Iters:', epoch)
            print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total), '\n')

            acc = 100. * float(correct) / float(total)
            acc_record.append(acc)

            print(f"Iters {epoch} Metrics for test data: accuracy = {accuracy:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, F1 = {f1:.4f}, MCC = {mcc:.4f}, MAE = {mae:.4f}", '\n')
        

        print(f'max_test_acc ={max_acc: .4f} at epoch {max_acc_epoch}')


        # save model
        print("\nSaving trained model state_dict ")
        snn.eval()
        path = f"Model_hp_cnn_snn_6layer_{n}.pt"
        torch.save(snn.state_dict(), path)

        model = SNN_Model()
        path_whole = f"Model_hp_cnn_snn_6layer_{n}w.pth"
        torch.save(model, path_whole)

        end_time_w = time.time()
        elapsed_time = end_time_w - start_time_w
        print(f"The operation took {elapsed_time} seconds.")

        # # Plotting the train_loss and acc
        # index = np.arange(len(acc_record))
        # plt.plot(index, acc_record, color='blue', label='test_acc')
        # plt.plot(index, loss_train_record, color='red', label='y values')

        # plt.title("Plot of train_loss and acc")
        # plt.xlabel("Epoch")
        # plt.grid(True)
        # plt.show()  


    # # Robustness Exp.
    # import skimage
    # for i in range(7):
    #     correct = 0.
    #     total = 0.
    #     Spike = 0.
    #     Spike2 = 0.
    #     noise = 'gaussian'
    #     with torch.no_grad():
    #         va = i * 0.01
    #         for batch_idx, (inputs, targets) in enumerate(test_loader):
    #             # 'gaussian','salt','pepper','s&p','speckle'
    #             if noise == 'gaussian':
    #                 inputsf = skimage.util.random_noise(inputs, mode='gaussian', clip=True, mean=0, var=1 * va)
    #             elif noise == 'salt':
    #                 inputsf = skimage.util.random_noise(inputs, mode='salt', clip=True, amount=va)
    #             elif noise == 'pepper':
    #                 inputsf = skimage.util.random_noise(inputs, mode='pepper', clip=True, amount=6 * va)
    #             elif noise == 'sp':
    #                 inputsf = skimage.util.random_noise(inputs, mode='s&p', clip=True, amount=2 * va)
    #             elif noise == 'speckle':
    #                 inputsf = skimage.util.random_noise(inputs, mode='speckle', clip=True, mean=0, var=10000 * va)

    #             inputsf = torch.from_numpy(inputsf)
    #             inputsf = inputsf.to(device)

    #             outputs, sumspike, sumspike2, _, _, _, _ = snn(input=inputsf, hebb1=hebb1, hebb2=hebb2,
    #                                                            wins=time_window)

    #             targets_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
    #             loss = criterion(outputs.cpu(), targets_)

    #             _, predicted = outputs.cpu().max(1)
    #             total += float(targets.size(0))
    #             correct += float(predicted.eq(targets).sum().item())
    #             Spike += sumspike.detach() / 8.0
    #             Spike2 += sumspike2.detach() / 8.0

    #         acc = 100. * float(correct) / float(total)
    #         Spikem = Spike.mean().cpu().numpy()
    #         Spikem2 = Spike2.mean().cpu().numpy()

    #     # print('Model:', names)
    #     print('{%s}||noise{%s}-i: %d || Test Accuracy : %.3f || Spike1:  %.3f || Spike2:  %.3f' % (
    #     names, noise, i, acc, Spikem, Spikem2))

    print('Complete..')




