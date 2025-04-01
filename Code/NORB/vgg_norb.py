'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np


class SRNLayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(SRNLayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
        torch.nn.init.xavier_normal_(self.weights,gain=0.01)
        self.exponent = exponent
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        self.padding = padding
#     @torch.compile
    def forward(self, y2,train_status=True):
        # print(self.weights.shape)
        # if train_status:
        #     Ef = (torch.rand_like(y2)>0.1).float()
        #     y2 = Ef*y2 
        # else:
        #     print('hofida')
        
        y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)
        # print(self.epsilon)
        
        y = y + self.epsilon*torch.randn_like(y)
        if self.normalize:
            GG = torch.std(y,dim=1)  
            # A =(GG<epsilon).float()*epsilon + (GG>=epsilon).float()*torch.std(y,dim=1)
            y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
            
        X = self.weights 
        X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)
        
        
        for i in range(self.degree-1):
            # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
            X = torch.concat((X, X[:,:,0:self.num_slices]**(i+2)),dim=2)
            # X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)
        
        # for i,j in self.iter:
        #     # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
        #     if i!=j:
        #         X = torch.concat((X, (X[:,:,i]*X[:,:,j]).unsqueeze(2)),dim=2)
        
        # G = X.view(X.shape[0],1,int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
        # X = nn.Unfold((3,3),padding=1)(G).permute(0,2,1)
        # # indices = torch.Tensor([2,4,6,8]).int()
        # X = X[:,:,3:7]
        
        X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)
        
        # M = X.pinverse()
        X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
        X_cov_inv = torch.linalg.inv(X_cov)
        M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
        
        W = torch.einsum('ijk,akb->aijb',M,y)
       
        pred_final = torch.einsum('bec,abcd->abed', X, W)   
        
        
        # y3 = y.unsqueeze(1).expand(-1,pred_final.shape[1],-1,-1)
        # print(y3.shape)
        err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
        
        err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
        # err = err/err.detach().max()
        if self.exponent:
            # print('here')
            A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
            # return torch.exp(-err)
            return A-0.5
        else:
            if self.normalize:
                return 1-err
            else:
                return -err

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16-SRN': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)
        if self.vgg_name == 'VGG16-SRN':
            self.srn1 = SRNLayer(2, 64, 5,padding=2,epsilon = 0.00001,num_slices=3,degree=1)

    def forward(self, x):
        if self.vgg_name == 'VGG16-SRN':
            x = self.srn1(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        if self.vgg_name=='VGG16-SRN':
            in_channels = 64
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# +
class Shallow_GMUCNN(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = 0.001, classes = 5,use_bn = True,dropping=0,poly_order_init=5):
        super(Shallow_GMUCNN, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.use_bn = True
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.convs = [] 
        self.dropping = 0 
        # print(kernels[0])
#         self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=3,degree=1,normalize=True)
        self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=0,epsilon = epsilons,num_slices=3,degree=1)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=0)

        self.mpool1 = nn.AvgPool2d(2)
        self.mpool2 = nn.AvgPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)


        self.bnorm_fc = nn.BatchNorm2d(128)
        self.fc1 = nn.Conv2d(128,128,1)
        self.fc2 = nn.Conv2d(128,classes,1)

        self.feat_net = nn.Sequential(

            # self.conv1,
            self.mpool1,
            self.bn1,
            # nn.ReLU(inplace=True),

            # self.conv1,
            # self.bn1,
            # nn.ReLU(inplace=True),
            # self.mpool1,

            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            #
            self.conv3,
            self.bn3,
            nn.ReLU(inplace=True),
            self.mpool3,
            # #
            self.conv4,
            self.bn4,
            nn.ReLU(inplace=True),
            self.mpool4
            #
        )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
       
        
    def forward(self, x,bazinga=1):
        x = self.srn1(x)
        x = self.feat_net(x)
        x_errs = x
        
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
#         xm = self.drop(xm)
        
#         xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm


# +
class Shallow_CNN(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = 0.001, classes = 5,use_bn = True,dropping=0,poly_order_init=5):
        super(Shallow_CNN, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.use_bn = True
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = []  
        self.convs = [] 
        self.dropping = dropping
        # print(kernels[0])
#         self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=3,degree=1,normalize=True)
#         self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons,num_slices=3,degree=1)
        
        self.conv1 = nn.Conv2d(input_channels,64,5, padding=0)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=0)

        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)


        self.bnorm_fc = nn.BatchNorm2d(128)
        self.fc1 = nn.Conv2d(128,128,1)
        self.fc2 = nn.Conv2d(128,classes,1)

        self.feat_net = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.mpool1,

            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            #
            self.conv3,
            self.bn3,
            nn.ReLU(inplace=True),
            self.mpool3,
            # #
            self.conv4,
            self.bn4,
            nn.ReLU(inplace=True),
            self.mpool4
            #
        )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
       
        
    def forward(self, x,bazinga=1):
        x = self.feat_net(x)
        x_errs = x

        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        
#         xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm


# -

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
