import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer_smoothing_filter(nn.Module):

    def __init__(self, num_layer, num_classes, scale, epoch, gpu, constant=False):

        super(Layer_smoothing_filter, self).__init__()
        self.scale = scale
        self.epoch = epoch
        self.num_classes = num_classes
        self.gpu = gpu
        self.constant = constant
        self.num_layer = num_layer
        self.net = nn.Linear(num_classes, num_classes, bias=False)

        layer = []
        for _ in range(self.num_layer):
            layer.append(self.net)
            #layer.append(nn.Softshrink(lambd=self.scale))

        self.filter = nn.Sequential(*layer)

    def put_weight(self, weight, num_classes):
        C = torch.ones(num_classes) - num_classes * torch.eye(num_classes)
        T = torch.diagflat(weight)
        identity = torch.eye(num_classes)
        if self.gpu:
            C, T, identity = C.cuda(), T.cuda(), identity.cuda()
       
        weight_matrix = identity + 1 / (num_classes - 1) * torch.matmul(T, C)
        #weight_matrix = identity + 1 / (num_classes - 1) * torch.matmul(C, T)

        self.net.weight.data = weight_matrix
        self.net.weight.requires_grad = False
        if not self.constant:
            return weight_matrix

    def homotopy_tanh(self, x):
        return torch.tanh(x) + self.epoch/100 * (x - torch.tanh(x))

    def forward(self, input, residual=None):
        if residual is None:
            residual=input
        #class_residual = torch.sum(torch.abs(residual), dim=0)
        #normalized_residual = (class_residual - torch.mean(class_residual)) / torch.std(class_residual)
        output = []

        if self.scale == 0:
            output = residual    
        elif self.constant:
            MyWeight = self.scale * torch.tensor([1.]*self.num_classes).cuda()
            self.put_weight(MyWeight, self.num_classes)
            output = self.filter(residual)  
        else:   
            for i in range(input.size(0)):
                residual_sample = residual[i]
                residual_abs    = torch.abs(residual_sample)
                normalized_residual = (residual_abs - torch.mean(residual_abs)) / torch.std(residual_abs)

                MyWeight = self.scale * torch.sigmoid(normalized_residual)                                                # sigmoid
                #MyWeight = self.scale * 1/(1 + torch.exp(-0.5*normalized_residual))

                residual_smoothed = torch.mv(self.put_weight(MyWeight, self.num_classes), residual_sample)
                output.append(residual_smoothed)

            output = torch.stack(output)
        
        #output = self.homotopy_tanh(output)          # nonlinear activation
        return output 

def get_residual(output, target, num_classes):
    batchsize = target.size()[0]
    one_hot = torch.zeros(batchsize, num_classes)
    one_hot[torch.arange(batchsize), target] = 1
    one_hot = one_hot.cuda()
    
    residual = torch.abs(output - one_hot)
    #residual = output - one_hot
    return residual

def onehot_smoothing(target, num_classes, eps):
    batchsize = target.size()[0]
    one_hot = torch.zeros(batchsize, num_classes).cuda()
    one_hot[torch.arange(batchsize), target] = 1 - num_classes / (num_classes - 1) *eps
    one_hot = one_hot + eps/(num_classes - 1) * torch.ones(batchsize, num_classes).cuda()  
    return one_hot

def Huber(input, mu):
    pad = nn.ReplicationPad2d(1)
    output = torch.sqrt((pad(input)[:,:,2:,1:-1] - input)**2 + (pad(input)[:,:,1:-1,2:] - input)**2 )
    l2 = output**2 / (2*mu) * torch.clone(output < torch.tensor(mu)).detach().float().cuda()
    l1 = (output - mu/2) * torch.clone(output >= torch.tensor(mu)).detach().float().cuda() 
    output = torch.sum(l1 + l2)

    return output 

class Weight_Filter(nn.Module):

    def __init__(self, scale, kernel_size=3):
        super(Weight_Filter, self).__init__()
        self.kernel_size = kernel_size
        self.scale = scale
        self.extension = int((self.kernel_size - 1)/2)
        self.padding = nn.ReflectionPad2d(self.extension)

    def compute_weight(self, input):
        normalized = (input - torch.mean(input)) / (torch.std(input) + 1e-10)
        return self.scale * torch.sigmoid(normalized)
        
    def forward(self, residual):
        residual_pad = self.padding(residual)
        output = torch.zeros(residual.size()).cuda()

        weight = self.compute_weight(residual).cuda()

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if (i == self.extension) and (j == self.extension):
                     output += (1-weight) * residual_pad[:, :, i:residual.size(2)+i, j:residual.size(3)+j]
                else:
                     output += weight / (self.kernel_size**2 - 1) * residual_pad[:, :, i:residual.size(2)+i, j:residual.size(3)+j]

        return output


class Weight_Network(nn.Module):

    def __init__(self, num_classes):
        super(Weight_Network, self).__init__()
        self.linear = nn.Linear(num_classes,num_classes)

        self._initialize_weights()

    def forward(self, residual):
        out = F.relu(self.linear(residual))
        out = F.softmax(out, dim=1)

        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
