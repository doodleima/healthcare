import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from torchsummary import summary

import numpy as np


### Patch & Positional Embedding
class swinEMB(nn.Module):
    def __init__(self, params):
        super(swinEMB, self).__init__()
        
        self.dim = int(params['dimension'])

        self.patch_size = int(params['patch_size']) 
        self.input_c = int(params['num_channels'])
        self.emb_size = int(params['emb_size']) 
        self.img_hwd = int(params['img_hwd'])

        self.dropout_rate = params['drop_rate']

        self.emb_pos = nn.Parameter(torch.empty((self.img_hwd//self.patch_size)**int(self.dim), self.emb_size))
        nn.init.kaiming_normal_(self.emb_pos, nonlinearity='relu') # replaced Weight Initialization code suggested by Kaiming He's


        if self.dim == 3: 
            self.conv_layer1 = nn.Conv3d(in_channels=self.input_c, out_channels=self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.flatten_layer1 = nn.Flatten(start_dim=2, end_dim=-1) 
        else: 
            self.conv_layer1 = nn.Conv2d(in_channels=self.input_c, out_channels=self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.flatten_layer1 = nn.Flatten(start_dim=1, end_dim=-1) 

        self.dropout_layer1 = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        b, c, h, w, d = x.shape 

        x = self.conv_layer1(x)
        x = self.flatten_layer1(x).transpose(1, 2)

        x += self.emb_pos
        x = self.dropout_layer1(x)
        
        return x


### Multi-head Self Attention >> must be modified
class swinMSA(nn.Module):
    def __init__(self, params, bias=True):
        super(swinMSA, self).__init__()

        self.emb_size = int(params['emb_size'])
        self.num_heads = int(params['num_heads'])
        self.dropout_rate = float(params['drop_rate'])

        self.Q = nn.Linear(self.emb_size, self.emb_size, bias=bias)
        self.K = nn.Linear(self.emb_size, self.emb_size, bias=bias)
        self.V = nn.Linear(self.emb_size, self.emb_size, bias=bias)
 
        self.drop_layer1 = nn.Dropout(self.dropout_rate)
        self.proj_layer1 = nn.Linear(self.emb_size, self.emb_size)


    def forward(self, x):
        Q = rearrange(self.Q(x), "b n (h d) -> b n h d", h=self.num_heads)
        K = rearrange(self.K(x), "b n (h d) -> b n h d", h=self.num_heads)
        V = rearrange(self.V(x), "b n (h d) -> b n h d", h=self.num_heads)

        scale = np.sqrt(self.emb_size)
        energy = torch.einsum('bhqd, hbkd -> bhqk', Q, K)

        att1 = F.softmax(energy, dim=2) / scale
        att2 = self.drop_layer1(att1)

        x = torch.einsum('bhal, bhlv -> bhav', att2, V)
        x = rearrange(x, 'b n h d -> b n (h d)')

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
        pass
        # x = self.linear_layer1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear_layer2(x)

        # return x


# class SwinBasicLayer(nn.Module):
#     '''
#     {SwinTransformerBlock, Downsample} * depth
#     SwinTransformerBlock(N % 2 == 1): -> shift_size = window_size // 2
#     Downsample: Patch Merging (Decreasing Number of Patches)
#     '''

    ### devide window for a different shape(shifted window)
    # def window_partition(x, window_size):
    #     x_shape = x.size()
    #     if len(x_shape) == 5: #3D shape
    #         b, d, h, w, c = x_shape
    #         # x.size = 8
    #         x = x.view(b, 
    #                    d // window_size[0], window_size[0], 
    #                    h // window_size[1], window_size[1], 
    #                    w // window_size[2], window_size[2], 
    #                    c)

    #         windows_total = window_size[0] * window_size[1] * window_size[2]
            
    #         windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, windows_total, c) # reshape >> no need contiguous()

    #     elif len(x_shape) == 4: #2D shape
    #         b, d, h, w = x_shape
    #         # x.size = 6
    #         x = x.view(b, h // window_size[0], window_size[0],
    #                 w // window_size[1], window_size[1], c)
            
    #         windows_total = window_size[0] * window_size[1]

    #         windows = x.permute(0, 1, 3, 2, 4, 5).contigous().view(-1, windows_total, c)

    #     return windows


    # def window_reverse(x, window_size, dims):
    #     if len(dims) == 4:
    #         b, d, h, w = dims
    #         x = x.view(b,
    #                    d // window_size[0], h // window_size[1], w // window_size[2],
    #                    window_size[0], window_size[1], window_size[2],
    #                    -1,)

    #         windows = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    #     elif len(dims) == 3:
    #         b, h, w = dims
    #         x = x.view(b, 
    #                    h // window_size[0], w // window_size[1], 
    #                    window_size[0], window_size[1], 
    #                    -1)
    #         windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    #     return windows

### N-Stack structure, input reshaping + windows partition + self attention layer
### Must be changed to a N-Stack loop form
class SwinTransformerBlock(nn.Module):
    def __init__(self, params):
        super(SwinTransformerBlock, self).__init__()
        
        self.dim = int(params['dimension'])
        self.patch_size = int(params['patch_size']) 

        self.input_c = int(params['num_channels'])
        self.emb_size = int(params['emb_size']) 
        self.img_hwd = int(params['img_hwd'])

        self.dropout_rate = params['drop_rate']

        self.hwd = self.img_hwd // self.patch_size
        self.windows = [8, 8, 8] # window size, need parameterize

        self.EMB = swinEMB(params) # Patch Embedding + Flatten Block
        self.MSA = swinMSA(params) # Multi-head Self Attention Block


    def window_partition(self, x, window_size): # window size must be gathered from the outside(N-Stack)
        if len(x.shape) == 5: #3D shape
            b, d, h, w, c = x.shape
            # x.shape = 8
            x = x.view(b, 
                       d // window_size[0], window_size[0], 
                       h // window_size[1], window_size[1], 
                       w // window_size[2], window_size[2], 
                       c)

        else: #2D shape
            b, d, h, w = x.shape
            # x.shape = 6
            x = x.view(b, h // window_size[0], window_size[0],
                    w // window_size[1], window_size[1], c)
            
        windows_total = window_size[0] * window_size[1] * window_size[2] if len(x.shape) == 5 else window_size[0] * window_size[1]
        windows = x.permute(0, 1, 3, 2, 4, 5).contigous().view(-1, windows_total, c)if len(x.shape) == 5 else x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, windows_total, c)

        return windows


    def window_reverse(self, x, window_size): # , dims
        if len(x.shape) == 4:
            b, d, h, w = x.shape
            x = x.view(b, d // window_size[0], h // window_size[1], w // window_size[2],
                          window_size[0], window_size[1], window_size[2], -1)

            windows = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

        elif len(x.shape) == 3:
            b, h, w = x.shape
            print(b, h, w)
            x = x.view(b, h // window_size[0], w // window_size[1], 
                          window_size[0], window_size[1], -1)

            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

        return windows


    def forward(self, x):
        x = self.EMB(x)
        x = self.window_partition(x.view(1, self.hwd, self.hwd, self.hwd, self.emb_size), self.windows) # change the value 1 to batch_size (future works)
        x = self.MSA(x)
        print(x.shape)
        x = self.window_reverse(x, self.windows)

        print(x.shape)
        return x


class PatchMerging(nn.Module): # Downsample
    def __init__(self, params):
        pass

    def forward(self, x):
        pass