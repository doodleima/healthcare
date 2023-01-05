import torch
import torch.nn as nn

from einops import rearrange, repeat


### Patch & Positional Embedding
class EMB(nn.Module):
    def __init__(self, params):
        super(EMB, self).__init__()
        
        self.patch_size = int(params['patch_size']) if params.get('patch_size') != None else 8 # default 16
        self.emb_size = int(params['emb_size']) if params.get('emb_size') != None else 4096
        self.input_c = int(params['num_channels'])
 
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size)) ### class token '+1'
        self.emb_pos = nn.Parameter(torch.randn((128//self.patch_size)**3+1, self.emb_size)) # hwd/p^3 +1, embedding size

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
        

    def forward(self, x):
        pass


### Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        # multi layer perceptron

    def forward(self, x):
        pass


### Transformer Encoder Block
class TF_enc(nn.Module):
    def __init__(self, params):
        super(TF_enc, self).__init__()
        # TransFormer Encoder Block

    def forward(self, x):
        pass



# class ViT(nn.Module): # or nn.Sequential(super class init)
#     def __init__(self, params):
#         super(TF_enc), self).__init__()
#         # TransFormer Encoder Block

#     def forward(self, x):
#         pass



if __name__ == '__main__':
    img = torch.randn([1, 1, 128, 128, 128])
    embedding = EMB(params={'num_channels':1, 'emb_size':4096, 'patch_size':8})
    z = embedding(img)
    print(z.size())