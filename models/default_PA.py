'''
The model definition code was modified based on the repository https://github.com/zhilin007/FFA-Net [1].
[1] X. Qin, Z. Wang, Y. Bai, X. Xie, H. Jia, Ffa-net: Feature fusion attention network for single image dehazing, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34, 2020, pp. 11908–11915.
'''


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU


class PyramidAttention(nn.Module):
    def __init__(self, level=3, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True):
        super(PyramidAttention, self).__init__()
        assert level == 3, 'Currently, only level = 3 is supported.'
        self.PALayer_base = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.PALayer_2x = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.PALayer_05x = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x_2x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # x_05x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_05x = F.interpolate(x, size=[x.size()[2]//2, x.size()[3]//2], mode='bilinear')
        att_base = self.PALayer_base(x)
        att_2x = self.PALayer_2x(x_2x)
        att_05x = self.PALayer_05x(x_05x)
        att_2x_recovery = F.interpolate(att_2x, scale_factor=0.5, mode='bilinear')
        # att_05x_recovery = F.interpolate(att_05x, scale_factor=2, mode='bilinear')
        att_05x_recovery = F.interpolate(att_05x, size=[x.size()[2], x.size()[3]], mode='bilinear')
        y = att_05x_recovery + att_base + att_2x_recovery
        return x * y

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)
    
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)#e.g., batchx2x1x64
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
        self.msa = PyramidAttention()
    def forward(self, x):
        res = self.gp(x)
        res = self.msa(res)
        res += x
        return res

class MODEL_PA(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(MODEL_PA, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        # assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        # self.msa = PyramidAttention()
        if(self.gps > 2):
            self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
        if(self.gps > 3):
            self.g4= Group(conv, self.dim, kernel_size,blocks=blocks)
        if(self.gps > 4):
            self.g5= Group(conv, self.dim, kernel_size,blocks=blocks)
        if(self.gps > 5):
            self.g6= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        # res2 = self.msa(res2)
        if(self.gps == 3):
            res3=self.g3(res2)
            w=self.ca(torch.cat([res1,res2,res3],dim=1))
            w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
            out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        elif(self.gps == 4):
            res3=self.g3(res2)
            res4=self.g4(res3)
            w=self.ca(torch.cat([res1,res2,res3,res4],dim=1))
            w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
            out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3+w[:,3,::]*res4
        elif(self.gps == 5):
            res3=self.g3(res2)
            res4=self.g4(res3)
            res5=self.g5(res4)
            w=self.ca(torch.cat([res1,res2,res3,res4,res5],dim=1))
            w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
            out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3+w[:,3,::]*res4+w[:,4,::]*res5
        elif(self.gps == 6):
            res3=self.g3(res2)
            res4=self.g4(res3)
            res5=self.g5(res4)
            res6=self.g6(res5)
            w=self.ca(torch.cat([res1,res2,res3,res4,res5,res6],dim=1))
            w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
            out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3+w[:,3,::]*res4+w[:,4,::]*res5+w[:,5,::]*res6
        out=self.palayer(out)
        x=self.post(out)
        return x + x1
if __name__ == "__main__":
    net=MODEL_PA(gps=3,blocks=19)
    print(net)