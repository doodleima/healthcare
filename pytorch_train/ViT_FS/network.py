import torch
import torch.nn as nn


# input parameter: Dictionary Type



### CDB Common Structure(Attr)
class CDB_struc(nn.Module):
    def __init__(self, params):
        super(CDB_struc, self).__init__()
        
        assert int(params['dimension']) != 2 or int(params['dimension']) != 3, 'Wrong Dimension value, please fill out 2 or 3'
        self.dim = int(params['dimension'])

        pad_w = int((params['kernel_width']-1) / 2)
        pad_h = int((params['kernel_height']-1) / 2)
        kernel_w = int(params['kernel_width']) 
        kernel_h = int(params['kernel_height']) 

        conv_input = int(params['num_channels'])
        conv_input2 = int(params['num_filters'])
        stride = int(params['stride_c'])

        # change activation function if sliu == True
        self.relu = nn.PReLU() if params['silu'] == False else nn.SiLU()

        if self.dim == 3: # 3d
            pad_d = int((params['kernel_depth']-1) / 2)
            kernel_d = int(params['kernel_depth']) 

            self.conv_layer1 = nn.Conv3d(in_channels=conv_input, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h, kernel_d), stride=stride, padding=(pad_w, pad_h, pad_d))
            self.conv_layer2 = nn.Conv3d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h, kernel_d), stride=stride, padding=(pad_w, pad_h, pad_d))
            self.conv_layer3 = nn.Conv3d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(1, 1, 1), stride=stride) # no padding

            self.bn0 = nn.BatchNorm3d(num_features=conv_input)
            self.bn1 = nn.BatchNorm3d(num_features=conv_input2)
            self.bn2 = nn.BatchNorm3d(num_features=conv_input2)

        else: # 2d
            self.conv_layer1 = nn.Conv2d(in_channels=conv_input, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h), stride=stride, padding=(pad_w, pad_h))
            self.conv_layer2 = nn.Conv2d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(kernel_w, kernel_h), stride=stride, padding=(pad_w, pad_h))
            self.conv_layer3 = nn.Conv2d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=(1, 1), stride=stride) # no padding

            self.bn0 = nn.BatchNorm2d(num_features=conv_input)
            self.bn1 = nn.BatchNorm2d(num_features=conv_input2)
            self.bn2 = nn.BatchNorm2d(num_features=conv_input2)

            
            
### CDB Basic Block
class CDB_Basic(CDB_struc):
    def __init__(self, params):
        super(CDB_Basic, self).__init__(params) # no params
        
        
    ## ActFunction(Previous Block) > [CONV - BN - (cat, max) - ActFunction]*2 - [CONV - BN] > OUTPUT
    def forward(self, x):
        dim = self.dim + 2

        # ActFunction(Previous Block)
        x0 = self.relu(x)
        
        # [CONV - BN - (cat, max) - ActFunction]*2
        x1 = self.conv_layer1(x0)
        x1_bn = torch.unsqueeze(self.bn0(x1), dim)
        x0_bn = torch.unsqueeze(x, dim)
        
        x2 = torch.cat((x1_bn, x0_bn), dim=dim)        # maxout
        x2_max, _ = torch.max(x2, dim)                 # maxout
        x2 = self.relu(x2_max)

        
        x2 = self.conv_layer2(x2)
        x2_bn = torch.unsqueeze(self.bn1(x2), dim)
        x1_bn = torch.unsqueeze(x, dim)
        
        x3 = torch.cat((x2_bn, x1_bn), dim=dim)        # maxout
        x3_max, _ = torch.max(x3, dim)                 # maxout
        x3 = self.relu(x3_max)

        # CONV - BN
        x_out = self.conv_layer3(x3)
        x_out = self.bn2(x_out)
        
        return x_out


    
### CDB Basic Block + ENC(Max Pooling Layer)
class CDB_Input(CDB_struc):
    def __init__(self, params):
        super(CDB_Input, self).__init__(params) # no params
        stride_pool = int(params['stride_p'])
        kernel_size = int(params['kernel_p'])
        conv_input2 = int(params['num_filters'])
    
        self.bn3 = nn.BatchNorm3d(num_features=conv_input2) if self.dim == 3 else nn.BatchNorm2d(num_features=conv_input2)
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride_pool, return_indices=True) if self.dim ==3 else nn.MaxPool2d(kernel_size=kernel_size, stride=stride_pool, return_indices=True)
        
        
    ## INPUT > BN > [CONV - BN - ActFunction] > [CONV - BN - (cat, max) - ActFunction] - [CONV - BN] > MAXPOOL > OUTPUT    
    def forward(self, x):        
        dim = self.dim + 2 # dimension 4(2d), 5(3d)

        # INPUT - BN
        x0_bn = self.bn0(x)

        # CONV - BN - ActFunction
        x0 = self.conv_layer1(x0_bn)
        x1_bn = self.bn1(x0)
        x1 = self.relu(x1_bn)

        x1 = self.conv_layer2(x1)
        x2_bn = self.bn2(x1)

        x2 = torch.cat((torch.unsqueeze(x2_bn, dim), torch.unsqueeze(x1_bn, dim)), dim=dim)
        x2_max, _ = torch.max(x2, dim)
        x2 = self.relu(x2_max)

        x3 = self.conv_layer3(x2)
        x_out = self.bn3(x3)
        
        out_enc, indices = self.maxpool(x_out)           # Max Pooling layer

        return x_out, out_enc, indices                   # indices to skip connection
    

    
### CDB Basic Block + ENC(Max pooling Layer)
class CDB_Enc(CDB_Basic):
    def __init__(self, params):
        super(CDB_Enc, self).__init__(params) # no params
        stride_pool = int(params['stride_p'])
        kernel_size = int(params['kernel_p'])
        
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride_pool, return_indices=True) if self.dim ==3 else nn.MaxPool2d(kernel_size=kernel_size, stride=stride_pool, return_indices=True)
        
        
    ## CDB Basic Block > ENC(Max Pooling Layer)
    def forward(self, x):        
        x_out = super(CDB_Enc, self).forward(x) # skip
        out_enc, indices = self.maxpool(x_out)
    
        return x_out, out_enc, indices
        
        
    
### DEC(Max Unpooling Layer) + CDB Basic Block
class CDB_Dec(CDB_Basic):
    def __init__(self, params):
        super(CDB_Dec, self).__init__(params) # no params
        stride_pool = int(params['stride_p'])
        kernel_size = int(params['kernel_p'])
        
        self.unpool = nn.MaxUnpool3d(kernel_size=kernel_size, stride=stride_pool) if self.dim ==3 else nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride_pool)
        
        
    ## INPUT > UNPOOL + SKIP CONNECTION > CDB Basic Block 
    def forward(self, x, x_out, indices):        
        dim = self.dim + 2 # dimension 4(2d), 5(3d)

        unpool = torch.unsqueeze(self.unpool(x, indices), dim) # unpooling layer
        x_out = torch.unsqueeze(x_out, dim) # output of the block(Enc)
        concatenated_max, _ = torch.max(torch.cat((unpool, x_out), dim=dim), dim)

        out_dec = super(CDB_Dec, self).forward(concatenated_max) # CDB Basic Block(Inherited from CDB_basic)
        
        return out_dec
        
            
### Bottle-Neck Block
class CDB_BottleNeck(nn.Module):
    def __init__(self, params): # outblock: removed 
        super(CDB_BottleNeck, self).__init__()
        
        assert int(params['dimension']) != 2 or int(params['dimension']) != 3, 'Wrong Dimension value, please fill out 2 or 3'
        self.dim = int(params['dimension'])
        
        pad_w = int((params['kernel_width']-1) / 2)
        pad_h = int((params['kernel_height']-1) / 2)
        kernel_w = int(params['kernel_width']) 
        kernel_h = int(params['kernel_height']) 
        
        conv_input = int(params['num_filters'])
        stride = int(params['stride_c'])
        
        # change activation function if sliu == True
        self.relu = nn.PReLU() if params['silu'] == False else nn.SiLU()

        if self.dim == 3: # 3d
            pad_d = int((params['kernel_depth']-1) / 2)
            kernel_d = int(params['kernel_depth']) 
            
            self.conv_layer1 = nn.Conv3d(in_channels=conv_input, out_channels=conv_input, kernel_size=(kernel_w, kernel_h, kernel_d), stride=stride, padding=(pad_w, pad_h, pad_d))
            self.conv_layer2 = nn.Conv3d(in_channels=conv_input, out_channels=conv_input, kernel_size=(kernel_w, kernel_h, kernel_d), stride=stride, padding=(pad_w, pad_h, pad_d))
            self.conv_layer3 = nn.Conv3d(in_channels=conv_input, out_channels=conv_input, kernel_size=(1, 1, 1), stride=stride) # no padding
            
            self.bn0 = nn.BatchNorm3d(num_features=conv_input)
            self.bn1 = nn.BatchNorm3d(num_features=conv_input)
            self.bn2 = nn.BatchNorm3d(num_features=conv_input)

        else: # 2d
            self.conv_layer1 = nn.Conv2d(in_channels=conv_input, out_channels=conv_input, kernel_size=(kernel_w, kernel_h), stride=stride, padding=(pad_w, pad_h))
            self.conv_layer2 = nn.Conv2d(in_channels=conv_input, out_channels=conv_input, kernel_size=(kernel_w, kernel_h), stride=stride, padding=(pad_w, pad_h))
            self.conv_layer3 = nn.Conv2d(in_channels=conv_input, out_channels=conv_input, kernel_size=(1, 1), stride=stride) # no padding

            self.bn0 = nn.BatchNorm2d(num_features=conv_input)
            self.bn1 = nn.BatchNorm2d(num_features=conv_input)
            self.bn2 = nn.BatchNorm2d(num_features=conv_input)


    ## [CONV - BN - ActFunction] > [CONV - BN - (cat, max) - ActFunction]*2 - [CONV - BN] > OUTPUT
    def forward(self, x):
        dim = self.dim + 2 # dimension 4(2d), 5(3d)

        # Activation from pooled input
        x0 = self.relu(x)

        # Convolution block 1
        x0 = self.conv_layer1(x0)
        x1_bn = self.bn0(x0)
        x0_bn = torch.unsqueeze(x, dim)
        x1_bn = torch.unsqueeze(x1_bn, dim)
        x1 = torch.cat((x1_bn, x0_bn), dim=dim)  # Concatenate along the 5th dimension NB x C x H x W x F
        x1_max, _ = torch.max(x1, dim)
        x1 = self.relu(x1_max)

        # Convolution block 2
        x1 = self.conv_layer2(x1)
        x2_bn = self.bn1(x1)
        x2_bn = torch.unsqueeze(x2_bn, dim)
        x1_max = torch.unsqueeze(x1_max, dim)
        x2 = torch.cat((x2_bn, x1_max), dim=dim)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, dim)
        x2 = self.relu(x2_max)

        # Convolution block 3
        out = self.conv_layer3(x2)
        out = self.bn2(out)
        
        return out
        
    
    
### Classifier Block
class CDB_Output(nn.Module):
    def __init__(self, params):
        super(CDB_Output, self).__init__()
        
        assert int(params['dimension']) != 2 or int(params['dimension']) != 3, 'Wrong Dimension value, please fill out 2 or 3'
        self.dim = int(params['dimension'])
        
        conv_input = int(params['num_channels'])
        num_classes = int(params['output_classes'])
        kernel_c = int(params['kernel_c'])
        stride = int(params['stride_c'])

        self.conv_layer = nn.Conv3d(conv_input, num_classes, kernel_c, stride) if self.dim == 3 else nn.Conv2d(conv_input, num_classes, kernel_c, stride)


    def forward(self, x):
        logits = self.conv_layer(x)

        return logits



class fsCNN(nn.Module):
    def __init__(self, params):
        super(fsCNN, self).__init__()
        self.dim = params['dimension']

        self.input = CDB_Input(params)        
        params['num_channels'] = params['num_filters']

        self.enc1 = CDB_Enc(params)
        self.enc2 = CDB_Enc(params)
        self.enc3 = CDB_Enc(params)
        self.bottle_neck = CDB_BottleNeck(params)
        
        self.dec1 = CDB_Dec(params)
        self.dec2 = CDB_Dec(params)
        self.dec3 = CDB_Dec(params)
        self.dec4 = CDB_Dec(params)
        
        self.output = CDB_Output(params)
        
        ### Net Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
       
        
    def forward(self, x):
        if self.dim == 2:
            skip_enc1, out_enc1, indice1 = self.input.forward(x)
            skip_enc2, out_enc2, indice2 = self.enc1.forward(out_enc1)
            skip_enc3, out_enc3, indice3 = self.enc2.forward(out_enc2)
            skip_enc4, out_enc4, indice4 = self.enc3.forward(out_enc3)
            
            bottle_neck = self.bottle_neck(out_enc4)
            
            out_dec4 = self.dec4.forward(bottle_neck, skip_enc4, indice4)
            out_dec3 = self.dec3.forward(out_dec4, skip_enc3, indice3)
            out_dec2 = self.dec2.forward(out_dec3, skip_enc2, indice2)
            out_dec1 = self.dec1.forward(out_dec2, skip_enc1, indice1)
            
            logits = self.output.forward(out_dec1)

        else: # dim 3
            skip_enc1, out_enc1, indice1 = self.input.forward(x)
            skip_enc2, out_enc2, indice2 = self.enc1.forward(out_enc1)
            skip_enc3, out_enc3, indice3 = self.enc2.forward(out_enc2)
            
            bottle_neck = self.bottle_neck(out_enc3)
            
            out_dec3 = self.dec4.forward(bottle_neck, skip_enc3, indice3)
            out_dec2 = self.dec3.forward(out_dec3, skip_enc2, indice2)
            out_dec1 = self.dec2.forward(out_dec2, skip_enc1, indice1)
            
            logits = self.output.forward(out_dec1)

        ### simple debug
        # print(x.shape)
        # print('-----------------ENC-----------------')
        # print(out_enc1.shape, out_enc2.shape)
        # print(out_enc3.shape, out_enc4.shape)
        
        # print('-----------------DEC-----------------')
        # print(out_dec4.shape, out_dec3.shape)
        # print(out_dec2.shape, out_dec1.shape)
        
        return logits