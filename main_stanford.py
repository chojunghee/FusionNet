# -----------------------------------------------------------------------------
# import packages
# -----------------------------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from torch.autograd import Variable

import sys, os, time, datetime
import numpy as np
import openpyxl
import argparse
import visdom
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# parameters from the argument 
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='FusionNet', type=str, help='choose a type of model')
parser.add_argument('--dataset', default='stanford', type=str, help='choose a dataset')
parser.add_argument('--data_path', default='../../data/stanford/', type=str, help='path to data directory')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--epochs', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr_initial', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--lr_final', default=1e-3, type=float, help='final learning rate ')
parser.add_argument('--weight_decay',   type=float, default=0.0)
parser.add_argument('--momentum',       type=float, default=0.0)
parser.add_argument('--result_dir', default='../result/', type=str, help='directory of test dataset')
parser.add_argument('--iteration', default=5, type=int, help='number of trials')
parser.add_argument('--moreInfo', default=None, type=str, help='add more information on trial to the visdom window')
parser.add_argument('--smoothing', default='off', type=str, help='smooth the residual')

args    = parser.parse_args()
model_name      = args.model_name
dataset		    = args.dataset
data_path       = args.data_path
batch_size      = args.batch_size
epochs          = args.epochs
lr_initial      = args.lr_initial
lr_final        = args.lr_final
weight_decay    = args.weight_decay
momentum        = args.momentum
iteration       = args.iteration
moreInfo        = args.moreInfo
smoothing       = args.smoothing

now             = datetime.datetime.now()
time_stamp      = now.strftime('%F_%H_%M_%S')
result_directory = args.result_dir + time_stamp + '_' + moreInfo

sys.path.insert(0, '../util')

from smoothing_util import *
from stanford_dataset import *
num_classes = 9

#-------------------------------------------------------------------------------
# smoothing parameters
#------------------------------------------------------------------------------
scale = torch.zeros(epochs)
index = torch.linspace(-5, 5, epochs)
b = torch.tensor(0.5)
mu = torch.tensor(2.5)
#scale = 1/(torch.cosh((index - mu)/(2*b))**2)                              # logistic
scale = torch.exp(-torch.abs(index - mu)/b)                                     # laplace
scale = scale * torch.clone(scale>torch.tensor(0.001)).detach().float()          # truncate the small scales 

# -----------------------------------------------------------------------------
# load dataset
# -----------------------------------------------------------------------------
bCuda = torch.cuda.is_available()

transform = transforms.Compose([
                transforms.ToTensor()       
            ])

set_train   = stanford_dataset(root = data_path, train = True, transform=transform)
set_test    = stanford_dataset(root = data_path, train = False, transform=transform)

# -----------------------------------------------------------------------------
# function for training the model
# -----------------------------------------------------------------------------
def train(epoch):
    # print('train the model at given epoch')

    loss_train = []

    model.train()
    
    if smoothing == 'on':
        #Filter = Layer_smoothing_filter(smoothing_number[epoch], num_classes, scale[epoch], epoch, gpu=bCuda).cuda()
        Filter = Weight_Filter(scale[epoch])

    for idx_batch, (input, label) in enumerate(loader_train):
        
        loss = 0
        loss_for_graph = 0

        for i in range(len(input)):
            # eliminate the negative label
            label[i] = (label[i]<0).long()*(num_classes-1) + (label[i]>=0).long()*label[i]

            if bCuda:
                data, target = input[i].unsqueeze(0).cuda(), label[i].unsqueeze(0).cuda()

            optimizer.zero_grad()
            output = model(data)
            
            if smoothing == 'on':
                loss += objective(Filter(residual), ans)
            else:
                loss += objective(output, target)

            loss_for_graph += objective(output, target)
        # loss = objective(output, target)
        # lsm     = F.log_softmax(output)
        # loss    = F.nll_loss(lsm, target)
        loss = loss / len(input)
        loss_for_graph = loss_for_graph / len(input)
        loss.backward()
        optimizer.step()
        loss_train.append(loss_for_graph.item())
        
    loss_train_mean = np.mean(loss_train)

    return {'loss_train_mean': loss_train_mean}
# -----------------------------------------------------------------------------
# function for testing the model
# -----------------------------------------------------------------------------
def test():
    # print('test the model at given epoch')

    test_result = []
    filename = []
    with torch.no_grad():
        model.eval()

        for idx_batch, (data, name) in enumerate(loader_test):

            for i in range(len(data)):
                if bCuda:
                    input = data[i].unsqueeze(0).cuda()

                output = model(input)
                pred   = output.data.max(1)[1]
                test_result.append(pred)       
                filename.append(name[i])

        return test_result, filename

loss_train = np.zeros((epochs, iteration))

# -----------------------------------------------------------------------------
# iteration for the training
# -----------------------------------------------------------------------------

for iter in range(iteration):

    print('%d-th trial' % (iter+1))

    loader_train    = torch.utils.data.DataLoader(set_train, batch_size = batch_size, shuffle = True, collate_fn=my_collate_train, drop_last = True)
    loader_test     = torch.utils.data.DataLoader(set_test, batch_size = batch_size, shuffle = False, collate_fn=my_collate_test, drop_last = True)

    # -----------------------------------------------------------------------------
    # load neural network model
    # -----------------------------------------------------------------------------
    from FusionNet import *
    from Unet import *

    if model_name == 'FusionNet':
        model = FusionNet(kernel=3, num_classes=num_classes, input_channels=3)
    elif model_name == 'Unet':
        model = Unet(kernel=3, num_classes=num_classes, input_channels=3)

    if bCuda:
        model = model.cuda()
        scale = scale.cuda()

    # -----------------------------------------------------------------------------
    # optimization algorithm
    # -----------------------------------------------------------------------------
    if(lr_initial == lr_final):

        optimizer   = optim.SGD(model.parameters(), lr = lr_initial, momentum = momentum, weight_decay = weight_decay)

    else:

        sys.path.insert(0, '../optimizer')
        from scheduler_learning_rate import *

        optimizer   = optim.SGD(model.parameters(), lr = lr_initial, momentum = momentum, weight_decay = weight_decay)
        #scheduler   = scheduler_learning_rate_sigmoid(optimizer, lr_initial, lr_final, epochs)
        scheduler   = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75], gamma=0.01)

    # Loss functions
    objective = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# print out the model information
# -----------------------------------------------------------------------------
# print(model)
# print(model.get_number_of_parameters())

    # -----------------------------------------------------------------------------
    # iteration for the epoch
    # -----------------------------------------------------------------------------
    for e in range(epochs):

        time_start  = time.time()

        if(lr_initial != lr_final):

            scheduler.step()

        result_train    = train(e)

        # At last epoch, test the model
        if e == epochs - 1:
            result_test, filename  = test()
            # save results
            os.makedirs(result_directory)
            cm = plt.cm.RdYlGn
            for i in range(len(result_test)):
                    im = result_test[i].squeeze().cpu().numpy()
                    plt.imsave(result_directory +'/result_'+ filename[i], im, cmap=cm)

        loss_train[e,iter]   = result_train['loss_train_mean']

        time_elapsed    = time.time() - time_start # in seconds

        for param_group in optimizer.param_groups:

            learning_rate = param_group['lr']

        print('epoch: {:3d}/{:3d}, lr: {:6.5f}, loss(mean): {:8.5f} ({:.0f}s) '.format(e+1, epochs, learning_rate, loss_train[e,iter], time_elapsed))
                
        #
        # early termination
        #
        #if (e + 1) > 20 and accuracy_test[e,iter] < 50: break


# ---------------------------------------------------------------------
# visdom plot
# ---------------------------------------------------------------------
env = dataset + '_' + model_name
vis = visdom.Visdom(port=2552, server='http://lk3.math.snu.ac.kr', env=env)

plt_title = '{}, {}, {}, epoch: {:d}, trial: {:d}, <br>batch size: {:d}, momentum: {:.1f}, weight decay: {:.5f}<br>learning rate: (i) {:.5f}, (f) {:.5f}'.format(
    args.dataset, model_name, moreInfo, epochs, iteration, batch_size, momentum, weight_decay, lr_initial, lr_final)

opts1 = dict(name='train loss',
             mode='lines',
             type='custom',
             marker={'color': 'blue', 'symbol': 'dot'})

layout = dict(title=plt_title,
              titlefont=dict(color='red', size='12'),
              xaxis={'title': 'epoch'},
              yaxis={'title': 'loss'})

# ---------------------------------------------------------------------
# initial trace
# ---------------------------------------------------------------------
trace1 = {'x': [], 'y': []}

loss_train_mean = np.mean(loss_train, axis=1)

trace1['x'] = [i for i in range(1, epochs+1)]
trace1['y'] = loss_train_mean.tolist()

trace1.update(opts1)

data = [trace1]
vis._send({'data': data, 'layout': layout, 'win': time_stamp})
vis.update_window_opts(win=time_stamp, opts=dict(width=900,height=300))

idx10           = max(int(epochs/10), 1)
trainloss_min   = np.min(loss_train_mean)
trainloss_last  = np.mean(loss_train_mean[-idx10:])

print(moreInfo)
print('loss(min): {:3.5f}'.format(trainloss_min))
print('loss(last): {:3.5f}'.format(trainloss_last))

def save_to_excel(path, data):
    if not os.path.isfile(path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['avg_trainloss'])
    else:
        wb = openpyxl.load_workbook(path)
        ws = wb.worksheets[0]
    for i in range(len(data)):
        ws.append([data[i]])
    ws.append(['trainloss_min','trainloss_last'])
    ws.append([trainloss_min, trainloss_last])
    wb.save(path)

filename = model_name + '_residual_'  + moreInfo + '_' + time_stamp + '.xlsx'
result = loss_train_mean
save_to_excel(args.result_dir + filename, result)
# torch.save(model.state_dict(), pathSaveModel)
