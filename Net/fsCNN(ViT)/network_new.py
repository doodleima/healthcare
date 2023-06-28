import torch
import torch.nn as nn

import os
import sys

p = os.path.abspath('..')
sys.path.insert(1, p)

from ViT_FS import vit

# input parameter: Dictionary Type
'''
    params = {'dimension': DIM, 
              'act_function': 'prelu' or 'mish', 
              'num_channels': 1, 
              'num_filters': 32, 
              'kernel_height': 3, 'kernel_width': 3, 
              'stride_c': 1, 'stride_p': 2,
              'kernel_c': 1, 'kernel_p': 2, 
              'output_classes': 5}

    [Common]
        dropout = decoder block

    [Dimension dependent]: 2D or 3D
        conv_layer1 = conv(3*3), input block only
        conv_layer2 = conv(3*3), encoder & decoder block
        conv_layer3 = conv(1*1), bottle-neck block only
        out_layer(conv_layer4) = output block(segmentation) only - fit the output channel as 'output_classes'

        bn0 = input & bottleneck block
        bn1 = encoder & decoder block

        maxpool = pooling layer after encoder block
        unpool = unpooling layer before decoder block
'''

### structure initiation
class struc_init(nn.Module):
    def __init__(self, params):
        super(struc_init, self).__init__()
        
        assert int(params['dimension']) != 2 or int(params['dimension']) != 3, 'Wrong Dimension value, please fill out 2 or 3'
        self.relu = nn.ReLU() if params['act_function'] == 'relu' else nn.Mish() # relu or mish
        self.dim = int(params['dimension'])

        pad_h = int((params['kernel_height']-1) / 2)
        pad_w = int((params['kernel_width']-1) / 2)
        kernel_h = int(params['kernel_height']) 
        kernel_w = int(params['kernel_width']) 
        conv_input = int(params['num_channels'])
        conv_input2 = int(params['num_filters'])
        kernel_c = int(params['kernel_c'])
        stride = int(params['stride_c'])
        kernel_size = int(params['kernel_p'])
        stride_pool = int(params['stride_p'])
        num_classes = int(params['output_classes'])

        self.dropout = nn.Dropout(float(params['drop_rate']))

        if self.dim == 3: # 3d
            pad_d = int((params['kernel_depth']-1) / 2)
            kernel_d = int(params['kernel_depth']) 

            self.conv_layer1 = nn.Conv3d(in_channels=conv_input, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h, kernel_d), stride=stride, padding=(pad_w, pad_h, pad_d)) # x(raw)            
            self.conv_layer2 = nn.Conv3d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h, kernel_d), stride=stride, padding=(pad_w, pad_h, pad_d))
            self.conv_layer3 = nn.Conv3d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(1, 1, 1), stride=stride) # no padding, conv 11

            self.bn0 = nn.BatchNorm3d(num_features=conv_input)
            self.bn1 = nn.BatchNorm3d(num_features=conv_input2)

            self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride_pool, return_indices=True)
            self.unpool = nn.MaxUnpool3d(kernel_size=kernel_size, stride=stride_pool)

            self.out_layer = nn.Conv3d(in_channels=conv_input2, out_channels=num_classes, kernel_size=kernel_c, stride=stride)

        else: # 2d
            self.conv_layer1 = nn.Conv2d(in_channels=conv_input, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h), stride=stride, padding=(pad_w, pad_h))
            self.conv_layer2 = nn.Conv2d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h), stride=stride, padding=(pad_w, pad_h))
            self.conv_layer3 = nn.Conv2d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(1, 1), stride=stride) # no padding, conv 11

            self.bn0 = nn.BatchNorm2d(num_features=conv_input)
            self.bn1 = nn.BatchNorm2d(num_features=conv_input2)

            self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride_pool, return_indices=True)            
            self.unpool = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride_pool)
    
            self.out_layer = nn.Conv2d(in_channels=conv_input2, out_channels=num_classes, kernel_size=kernel_c, stride=stride)



### Basic Block Strcuture
class Basic(struc_init):
    def __init__(self, params):
        super(Basic, self).__init__(params) # no params


    ## ActFunction(Previous Block) > [CONV - BN - (cat, max) - ActFunction]*2 - [CONV - BN] > OUTPUT
    def forward(self, x, is_input=False):
        dim = self.dim + 2

        ### input block only: [BN -> CONV(3*3) -> BN]
        if is_input == True:
            x1 = self.bn0(x)
            x1 =  self.conv_layer1(x1)
            x_out =  self.bn1(x1)

        ### [ACTFUNC -> CONV(3*3) -> BN]
        else:
            x1 = self.relu(x)
            x1 = self.conv_layer2(x1)
            x_out = self.bn1(x1)

        return x_out


    
### Input Block Structure 
class Enc_Input(Basic):
    def __init__(self, params):
        super(Enc_Input, self).__init__(params) # no params        
        

    ### [BN -> CONV(3*3) -> BN] -> [ACTFUNC -> CONV(3*3) -> BN] -> MAXOUT -> [ACTFUNC -> CONV(3*3) -> BN]    
    def forward(self, x):        
        dim = self.dim + 2 # dimension 4(2d), 5(3d)

        ### [BN -> CONV(3*3) -> BN] -> [ACTFUNC -> CONV(3*3) -> BN]
        x_in = super().forward(x, True)
        x1 = super().forward(x_in)

        ### [ACTFUNC -> CONV(3*3) -> BN]
        x1_out, _ = torch.max(torch.cat((torch.unsqueeze(x_in, dim), torch.unsqueeze(x1, dim)), dim=dim), dim)
        x2 = super().forward(x1_out)

        ### pooling layer
        x_out, indices = self.maxpool(x2)

        return x_out, indices    


    
### Encoder Block Structure
class Enc(Basic):
    def __init__(self, params):
        super(Enc, self).__init__(params) # no params


    ## [ACTFUNC -> CONV(3*3) -> BN] -> {MAXOUT -> [ACTFUNC -> CONV(3*3) -> BN]*3}
    def forward(self, x, is_bn = False):               
        dim = self.dim + 2 # dimension 4(2d), 5(3d)
        
        ### {[ACTFUNC -> CONV(3*3) -> BN] -> MAXOUT}*2
        x1 = super().forward(x)
        x1_out, _ = torch.max(torch.cat((torch.unsqueeze(x, dim), torch.unsqueeze(x1, dim)), dim=dim), dim)

        x2 = super().forward(x1_out)
        x2_out, _ = torch.max(torch.cat((torch.unsqueeze(x1_out, dim), torch.unsqueeze(x2, dim)), dim=dim), dim)

        ### [ACTFUNC -> CONV(3*3) -> BN]
        x_out = super().forward(x2_out)
        x_out, indices = self.maxpool(x_out)

        return x_out, indices
        
        
    
### Decoder Block Structure
class Dec(Basic):
    def __init__(self, params):
        super(Dec, self).__init__(params) # no params
        
    ## INPUT > UNPOOL + SKIP CONNECTION > CDB Basic Block 
    def forward(self, x, x_out_vit, indices, is_final=False):        
        dim = self.dim + 2 # dimension 4(2d), 5(3d)

        ### UNPOOL
        x0 = self.unpool(x, indices)
        x0_out = torch.unsqueeze(x0, dim)
        v1_out = torch.transpose(torch.unsqueeze(x_out_vit, dim), 0, 1) ### should be changed: Unpooling instead of transpose convolutions

        x1_out, _ = torch.max(torch.cat((x0_out, v1_out), dim=dim), dim)

        ### {[ACTFUNC -> CONV(3*3) -> BN]-> MAXOUT}*2
        x2 = super().forward(x1_out)
        x2_out, _ = torch.max(torch.cat((torch.unsqueeze(x1_out, dim), torch.unsqueeze(x2, dim)), dim=dim), dim)

        x3 = super().forward(x2_out)
        x3_out, _ = torch.max(torch.cat((torch.unsqueeze(x2_out, dim), torch.unsqueeze(x3, dim)), dim=dim), dim)

        ### [ACTFUNC -> CONV(3*3) -> BN]
        x_out = super().forward(x3_out)
        if is_final == True: x_out = self.dropout(x_out)

        return x_out
        


### Bottle-Neck Block Structure
class BN(Basic):
    def __init__(self, params): # outblock: removed 
        super(BN, self).__init__(params)
        # self.conv_layer2 = self.conv_layer3

    ## [CONV - BN - (cat, max) - ActFunction] - [CONV - BN - DropOut]
    def forward(self, x):
        dim = self.dim + 2 # dimension 4(2d), 5(3d)
        
        ### {[ACTFUNC -> CONV(1*1) -> BN] -> MAXOUT}*2
        x1 = super().forward(x)
        x1_out, _ = torch.max(torch.cat((torch.unsqueeze(x, dim), torch.unsqueeze(x1, dim)), dim=dim), dim)
                
        x2 = super().forward(x1_out)
        x2_out, _ = torch.max(torch.cat((torch.unsqueeze(x1_out, dim), torch.unsqueeze(x2, dim)), dim=dim), dim)

        ### [ACTFUNC -> CONV(1*1) -> BN]
        x_out = super().forward(x2_out)

        return x_out
        
    
    
### Output Block Structure
class Output(struc_init):
    def __init__(self, params):
        super(Output, self).__init__(params)

    def forward(self, x):
        return self.out_layer(x)

        
### Updated fsCNN
class fsCNN(nn.Module):
    def __init__(self, params):
        super(fsCNN, self).__init__()

        self.vit_out = vit.ViT(params) # ViT initiation

        self.input = Enc_Input(params)        
        self.enc= Enc(params)
        self.dec = Dec(params)
        self.bottle_neck = BN(params)
        self.output = Output(params)

        ### Net Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
        
    def forward(self, x):
        x0, x1, x2 = self.vit_out(x) #, x3
                    
        enc_out1, indice1 = self.input.forward(x)
        enc_out2, indice2 = self.enc.forward(enc_out1)
        enc_out3, indice3 = self.enc.forward(enc_out2)

        bottle_neck = self.bottle_neck(enc_out3)
                    
        dec_out3 = self.dec.forward(bottle_neck, x2, indice3)
        dec_out2 = self.dec.forward(dec_out3, x1, indice2)
        dec_out1 = self.dec.forward(dec_out2, x0, indice1)

        logits = self.output.forward(dec_out1)

        return logits