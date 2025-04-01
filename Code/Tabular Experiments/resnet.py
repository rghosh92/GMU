'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockMixed(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockMixed, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x[0])))
        out = self.bn2(self.conv2(out))
        x1_forward = self.shortcut(x[1])
        out += x1_forward
        out = F.relu(out)
        return out,x1_forward


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResMLPNet(nn.Module):
    def __init__(self, input_channels, block, num_blocks, num_classes=10):
        super(ResMLPNet, self).__init__()
        self.in_planes = 512 
        self.conv1 = nn.Conv2d(input_channels, 512, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.layer1 = self._make_layer(block, 512, num_blocks[0])
        self.layer2 = self._make_layer(block, 512, num_blocks[1])
        self.linear = nn.Linear(512, num_classes)
        
        
        self.drop = nn.Dropout(0)
        
        
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn1(self.conv1(x))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.drop(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,0

class GMULayer(nn.Module):
    def __init__(self,input_channels, output_channels,medians, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(GMULayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        self.medians = medians.squeeze()
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        self.hada_mult = torch.nn.Parameter(torch.zeros(1,output_channels,input_channels))
        # torch.nn.init.xavier_uniform_(self.weights,gain=0.01)
        self.exponent = exponent
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        
        self.init_weights()
        
    def init_weights(self):
        # torch.nn.init.xavier_normal_(self.weights)
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0.5)
        

    def forward(self, y,train_status):
            # print(self.weights.shape)
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
            
            y = y + self.epsilon*torch.rand_like(y)
            
           
            y = y.unsqueeze(1).repeat(1,self.weight_bias.shape[0],1)
            y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
           
            
            if self.normalize == True:
                y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,self.weights.shape[1])
            X = self.weights
            for i in range(self.degree-1):
                X = torch.concat((X, self.weights**(i+2)),dim=2)
            
            
            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
           
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            err = torch.mean((y-pred_final)**2,dim=2)
            err = err.unsqueeze(2).unsqueeze(3)
            
            
            return torch.sqrt(1.000001-err)
            


class SimpleGMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(SimpleGMULayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        self.exponent = exponent
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        
        self.init_weights()
        
    def init_weights(self):
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0.5)
        

    def forward(self, y,p):
            # print(self.weights.shape)
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
           
            y = y + self.epsilon*torch.rand_like(y)
           
            y = y.unsqueeze(1).repeat(1,self.weight_bias.shape[0],1)
            y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
            
            if self.num_slices == 0:
                err = torch.mean((y)**2,dim=2)
                err = err.unsqueeze(2).unsqueeze(3)
                return (torch.exp(-err))
            
            if self.normalize == True:
                y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,self.weights.shape[1])
            
            X = self.weights
            for i in range(self.degree-1):
                X = torch.concat((X, self.weights**(i+2)),dim=2)
             
            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            err = torch.mean((y-pred_final)**2,dim=2)
            err = err.unsqueeze(2).unsqueeze(3)
           
            return torch.exp(-err)
#             








class SimpleGMULayerv2(nn.Module):
    def __init__(self,input_channels, output_channels, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(SimpleGMULayerv2, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        self.hada_mult = torch.nn.Parameter(torch.ones(1,output_channels,input_channels))
        self.exponent = exponent
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        self.threshold = torch.nn.Parameter(0.00001*torch.ones(1))
        
        self.init_weights()
        
    def init_weights(self):
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0.5)
        

    def forward(self, y,p):
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
           
            y = y + self.epsilon*torch.rand_like(y)
           
            y = y.unsqueeze(1).repeat(1,self.weight_bias.shape[0],1)

            
            if self.num_slices == 0:
                y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
                err = torch.mean((y)**2,dim=2)
                err = err.unsqueeze(2).unsqueeze(3)
                return (torch.exp(-err)),err
                
            if self.normalize == True:
                y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,self.weights.shape[1])
           
            X = self.weights
            for i in range(self.degree-1):
                X = torch.concat((X, self.weights**(i+2)),dim=2)
             
            X = torch.concat((torch.ones((X.shape[0],X.shape[1],1)), X),dim=2)

            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
           
            
            W = torch.einsum('ijk,bik->ijb',M,y)
           
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            err = torch.mean((y-pred_final)**2,dim=2)
            err = err.unsqueeze(2).unsqueeze(3)
            
           
                
            return torch.exp(-err)
                
            
           

class ResGMUMLP(nn.Module):
    def __init__(self, input_channels, block, num_blocks,medians,num_slices = 1, degree = 1, normalize = True, epsilon=0.0001, num_classes=10,use_dropout=True):
        super(ResGMUMLP, self).__init__()
        madden = 512 
        self.in_planes = madden
        projections = 512
        self.gmu1 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 0,degree=degree,normalize=normalize)
        self.gmu2 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 1,degree=degree,normalize=normalize)
        self.gmu3 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 2,degree=degree,normalize=normalize)
        self.gmu4 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 3,degree=degree,normalize=normalize)
   
        self.bn1 = nn.BatchNorm2d(madden)
        self.bn2 = nn.BatchNorm2d(512)
        self.use_dropout = use_dropout
        self.layer1 = self._make_layer(block, 512, num_blocks[0])
        self.layer2 = self._make_layer(block, 512, num_blocks[1])
        self.linear = nn.Linear(512, num_classes)
        # self.linear2 = nn.Linear(512, num_classes)
        
        
        self.drop = nn.Dropout(0.2)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        x_errs1 = self.gmu1(x,self.training)
    
        x_errs2 = self.gmu2(x,self.training)
        x_errs3 = self.gmu3(x,self.training)
        x_errs4 = self.gmu4(x,self.training)
        x_errs = torch.hstack([x_errs1,x_errs2,x_errs3,x_errs4])
        out = self.bn1(x_errs)
        
        out = self.layer1(out)
        out = self.layer2(out)
        if self.use_dropout:
            out = self.drop(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,x_errs





class SimpleGMUMLP(nn.Module):
    def __init__(self, input_channels,num_slices = 1, degree = 1, normalize = True, epsilon=0.0001, num_classes=10,use_dropout=True):
        super(SimpleGMUMLP, self).__init__()
        madden = 512 
        # self.conv1 = nn.Conv2d(input_channels, projections, 1)
        # self.gmu1 = MaskGMULayer(input_channels, madden,epsilon = epsilon,num_slices = num_slices,degree=degree,normalize=normalize)
        # print('abbaba')
        self.gmu1 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 0,degree=degree,normalize=normalize)
        self.gmu2 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 1,degree=degree,normalize=normalize)
        self.gmu3 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 2,degree=degree,normalize=normalize)
        self.gmu4 = SimpleGMULayer(input_channels, 128,epsilon = epsilon,num_slices = 3,degree=degree,normalize=normalize)
        
        self.use_dropout = use_dropout
        # self.recon1 = nn.Linear(madden, madden)
        self.bn1 = nn.BatchNorm2d(madden)
        self.bn2 = nn.BatchNorm1d(madden)
        self.fc = nn.Linear(madden, madden)
        
        self.linear = nn.Linear(madden, num_classes)
        
        
        
        self.drop = nn.Dropout(0)

    def forward(self, x,p=1):
        # x = self.conv1(x)
        x_errs1 = self.gmu1(x,self.training)
        x_errs2 = self.gmu2(x,self.training)
        x_errs3 = self.gmu3(x,self.training)
        x_errs4 = self.gmu4(x,self.training)
#         print(x_errs1.shape)
        x_errs = torch.hstack([x_errs1,x_errs2,x_errs3,x_errs4])
        out = self.bn1(x_errs)
        if self.use_dropout:
            out = self.drop(out)
        # out = F.avg_pool2d(out, 4)
        feats = out.view(out.size(0), -1)
        out = self.linear(F.relu(self.bn2(self.fc(feats))))
        
        return out,x_errs

class SimpleMLP(nn.Module):
    def __init__(self, input_channels,layers=[512],num_classes = 10,use_dropout=True):
        super(SimpleMLP, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, projections, 1)
        # self.gmu1 = MaskGMULayer(input_channels, madden,epsilon = epsilon,num_slices = num_slices,degree=degree,normalize=normalize)
        self.conv1 = nn.Conv2d(input_channels, layers[0],1)
        
        self.use_dropout = use_dropout
        # self.recon1 = nn.Linear(madden, madden)
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.linear = nn.Linear(layers[-1], num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.conv1(x)
        # x = x*np.sqrt(x.shape[1])/(torch.norm(x,dim=1)).unsqueeze(1).repeat(1,x.shape[1],1,1)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        if self.use_dropout:
            out = self.drop(out)
        # out = F.avg_pool2d(out, 4)
        feats = out.view(out.size(0), -1)
        out = self.linear(feats)
        
        return out,0


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
