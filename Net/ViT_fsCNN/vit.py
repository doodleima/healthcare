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


### Transformer Encoder Block(Basic)
class Enc_basic(nn.Module):
    def __init__(self, params):
        super(Enc_basic, self).__init__()
        # TransFormer Encoder Block

        self.emb_size = int(params['emb_size'])
        self.num_heads = int(params['num_heads'])
        self.dropout_rate = float(params['drop_rate'])

        self.residual_layer1 = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            MSA(params),
            nn.Dropout(self.dropout_rate)
        )

        self.residual_layer2 = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            MLP(params),
            nn.Dropout(self.dropout_rate)
        )


    def forward(self, x): ### if) multi-batch > n2 patch
        res = x
        x1 = self.residual_layer1(x)
        x1 += res

        res2 = x1
        x2 = self.residual_layer2(x1)
        x2 += res2

        return x2


### Transformer Encoder Block(Basic)
class TF_enc(nn.Module):
    def __init__(self, params):
        super().__init__()
        num_layers = params['num_layers']

        self.emb_layer1 = EMB(params)
        self.blocks = nn.ModuleList([Enc_basic(params) for _ in range(num_layers)])


    def forward(self, x):
        x_out = []
        x = self.emb_layer1(x)

        for blocks in self.blocks: x_out.append(blocks(x))

        return x_out


### ViT(Non-classification layer)
class ViT(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        params['num_layers'] = params['num_layers'] if params['num_layers'] else 4
        self.patch_size = params['patch_size']
        
        self.patch = self.patch_size*2
        self.patch1 = self.patch_size
        self.patch2 = int(self.patch_size/2) 
        self.patch3 = int(self.patch_size/4) 

        self.img_size = (params['img_hwd'], params['img_hwd'], params['img_hwd'])
        self.emb_size = params['emb_size']
        self.proj_axis = (0, 3 + 1) + tuple(d + 1 for d in range(3)) # 3D
        
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, (self.patch, self.patch, self.patch)))
        self.proj_reshape = list(self.feat_size) + [self.emb_size]
        
        self.input_c = int(params['num_channels']) # N2C

        self.tf_enc_layer1 = TF_enc(params)

        self.deconv_layer1 = nn.ConvTranspose3d(in_channels=self.emb_size, out_channels=self.input_c, kernel_size=self.patch, stride=self.patch)
        self.deconv_layer2 = nn.ConvTranspose3d(in_channels=self.emb_size, out_channels=self.input_c, kernel_size=self.patch1, stride=self.patch1)
        self.deconv_layer3 = nn.ConvTranspose3d(in_channels=self.emb_size, out_channels=self.input_c, kernel_size=self.patch2, stride=self.patch2)
        self.deconv_layer4 = nn.ConvTranspose3d(in_channels=self.emb_size, out_channels=self.input_c, kernel_size=self.patch3, stride=self.patch3)

        ### use another 3d reshape method instead of ConvTranspose3d
        

    def proj_feat(self, x):
        new_shape = [x.size()[0]] + self.proj_reshape
        x = x.view(new_shape)
        x = x.permute(self.proj_axis).contiguous()

        return x      


    def forward(self, x): 
        x_out = self.tf_enc_layer1(x)
        middle = int(len(x_out)/4)

        ### each output enters into the skip-connection layer(concatenate with the previous decoder output) 
        x_out_raw = self.proj_feat(x_out[0])
        x_out1 = self.proj_feat(x_out[int(middle*2)])
        x_out2 = self.proj_feat(x_out[int(middle*3)])
        x_out3 = self.proj_feat(x_out[-1])

        ### deconvolution
        ### b, num_filters, h, w, d ex) 1, 32, 128, 128, 128
        x_out_raw = self.deconv_layer1(x_out_raw).transpose(0, 1)
        x_out1 = self.deconv_layer2(x_out1).transpose(0, 1)
        x_out2 = self.deconv_layer3(x_out2).transpose(0, 1)
        x_out3 = self.deconv_layer4(x_out3).transpose(0, 1)

        return x_out_raw, x_out1, x_out2, x_out3 