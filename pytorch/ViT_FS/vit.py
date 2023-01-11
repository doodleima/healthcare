import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


### Patch & Positional Embedding
class EMB(nn.Module):
    def __init__(self, params):
        super(EMB, self).__init__()
        
        self.patch_size = int(params['patch_size']) #if params.get('patch_size') != None else 8 # default 16
        self.input_c = int(params['num_channels'])
        self.emb_size = int(params['emb_size']) #if params.get('emb_size') != None else 4096 (128, 128, 128)
        self.img_hwd = int(params['img_hwd'])

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size)) ### class token '+1'
        self.emb_pos = nn.Parameter(torch.randn((self.img_hwd//self.patch_size)**3+1, self.emb_size)) # hwd/p^3 +1, embedding size

        self.conv_layer1 = nn.Conv3d(in_channels=self.input_c, out_channels=self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)


    def forward(self, x):
        b, c, h, w, d = x.shape # 5D

        x = self.conv_layer1(x)
        x = rearrange(x, 'b e (h) (w) (d) -> b (h w d) e')  # x = x.reshape(b, -1, self.emb_size).transpose(1, 2)
        
        cls_token = repeat(self.cls_token, "() n e -> b n e", b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.emb_pos
        
        return x


### Multi-head Self Attention
class MSA(nn.Module):
    def __init__(self, params):
        super(MSA, self).__init__()

        self.emb_size = int(params['emb_size'])
        self.num_heads = int(params['num_heads'])
        self.dropout_rate = float(params['drop_rate'])

        self.K = nn.Linear(self.emb_size, self.emb_size)
        self.Q = nn.Linear(self.emb_size, self.emb_size)
        self.V = nn.Linear(self.emb_size, self.emb_size)
 
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
        self.emb_size = int(params['emb_size']) #if params.get('emb_size') != None else 4096
        self.dropout_rate = float(params['drop_rate']) #if params.get('drop_rate') != None else 0.2

        self.linear_layer1 = nn.Linear(self.emb_size, self.expansion * self.emb_size)
        self.linear_layer2 = nn.Linear(self.expansion * self.emb_size, self.emb_size)
        self.relu = nn.SiLU() #nn.GELU() if params['silu'] == False else nn.SiLU()
        self.dropout = nn.Dropout()


    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)

        return x


# ### Residual Block
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn_layer = fn

#     def forward(self, x, **kwargs):
#         x_res = x
#         x_out = self.fn_layer(x, **kwargs)
#         x_out += x_res

#         return x


### Transformer Encoder Block
class TF_enc(nn.Module):
    def __init__(self, params):
        super(TF_enc, self).__init__()
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

    def forward(self, x):
        res = x
        x1 = self.residual_layer1(x)
        x1 += res

        res2 = x1
        x2 = self.residual_layer2(x1)
        x2 += res2

        return x2


class ViT(nn.Module): # or nn.Sequential(super class init)
    def __init__(self, params):
        super(ViT), self).__init__()

        self.embbed = EMB(params)
        self.enc = TF_enc(params)

    def forward(self, x):
        pass


if __name__ == '__main__':
    h, w, d = 128, 128, 128
    patch_size = h/8
    emb_size = (h/patch_size)**3
    
    img = torch.randn([1, 1, h, w, d])
    params = {'num_channels':1, 'img_hwd':h, 'emb_size':emb_size, 'patch_size':patch_size, 'num_heads':32, 'drop_rate':0.2}
    embedding = EMB(params=params)
    attention = MSA(params=params)
    perceptron = MLP(params=params)
    enc = TF_enc(params=params)

    z = embedding(img)
    print(z.size())
    y = attention(z)
    print(y.size())
    x = perceptron(y)
    print(x.size())
    w = enc(x)
    print(w.size())