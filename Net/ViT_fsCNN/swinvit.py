import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from torchsummary import summary

import numpy as np


### Patch & Positional Embedding
class EMB(nn.Module):
    def __init__(self, params):
        super(EMB, self).__init__()
        
        self.patch_size = int(params['patch_size']*2) 
        self.input_c = int(params['num_channels'])
        self.emb_size = int(params['emb_size']) 
        self.img_hwd = int(params['img_hwd'])

        self.dropout_rate = params['drop_rate']

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size)) ### class token '+1'
        self.emb_pos = nn.Parameter(torch.randn((self.img_hwd//self.patch_size)**3, self.emb_size)) # hwd/p^3 '+1', embedding size

        self.conv_layer1 = nn.Conv3d(in_channels=self.input_c, out_channels=self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.dropout_layer1 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        b, c, h, w, d = x.shape # 5D

        x = self.conv_layer1(x)
        x = rearrange(x, 'b e (h) (w) (d) -> b (h w d) e')  # x = x.reshape(b, -1, self.emb_size).transpose(1, 2)
        
        x += self.emb_pos
        x = self.dropout_layer1(x)  # added
        
        return x


### Multi-head Self Attention
class MSA(nn.Module):
    def __init__(self, params):
        super(MSA, self).__init__()

        self.emb_size = int(params['emb_size'])
        self.num_heads = int(params['num_heads'])
        self.dropout_rate = float(params['drop_rate'])

        self.K = nn.Linear(self.emb_size, self.emb_size)#, self.emb_size)
        self.Q = nn.Linear(self.emb_size, self.emb_size)#, self.emb_size)
        self.V = nn.Linear(self.emb_size, self.emb_size)#, self.emb_size)
 
        self.drop_layer1 = nn.Dropout(self.dropout_rate)
        self.proj_layer1 = nn.Linear(self.emb_size, self.emb_size)


    def forward(self, x, mask: torch.Tensor=None):
        Q = rearrange(self.Q(x), "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(self.K(x), "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(self.V(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, hbkd -> bhqk', Q, K)
        if mask != None: energy.mask_fill(~mask, torch.finfo(torch.float32).min)

        scale = self.emb_size ** (1/2)
        att1 = F.softmax(energy, dim=1) / scale
        att2 = self.drop_layer1(att1)

        x = torch.einsum('bhal, bhlv -> bhav', att2, V)
        x = rearrange(x, 'b h n d -> b n (h d)')

        out = self.proj_layer1(x)

        return out


### Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()

        self.expansion = 4        
        self.emb_size = int(params['emb_size'])
        self.dropout_rate = float(params['drop_rate'])

        self.linear_layer1 = nn.Linear(self.emb_size, self.expansion * self.emb_size)
        self.linear_layer2 = nn.Linear(self.expansion * self.emb_size, self.emb_size)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout()


    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)

        return x


### devide window for a different shape(shifted window)
def window_partition(x, window_size):
    x_shape = x.size()
    if len(x_shape) == 5: #3D shape
        b, d, h, w, c = x_shape
        # x.size = 8
        x = x.view(b, d // window_size[0], window_size[0], 
                   h // window_size[1], window_size[1], 
                   w // window_size[2], window_size[2], c)

        windows_total = window_size[0] * window_size[1] * window_size[2]
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, windows_total, c) # reshape >> no need contiguous()

    elif len(x_shape) == 4: #2D shape
        b, d, h, w = x_shape
        # x.size = 6
        x = x.view(b, h // window_size[0], window_size[0],
                   w // window_size[1], window_size[1], c)
        
        windows_total = window_size[0] * window_size[1]
        windows = x.permute(0, 1, 3, 2, 4, 5).contigous().view(-1, windows_total, c)

    return windows