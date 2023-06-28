import torch
import torch.nn as nn


### input parameter: Dictoinary Type
'''
    params = {
              'batch_size': 4,
              'img_size': int = 128,
              'dimension': int = 2(3), 
              'act_function': str = 'mish', 
              'num_channels': int = 1, 
              'num_filters': int = 32, 
              'kernel_height': int = 3, 'kernel_width': int = 3, 
              'stride_c': int = 1, 'stride_p': int = 2,
              'kernel_c': int = 1, 'kernel_p': int = 2, 
              'output_classes': int = 5
              }
'''


### initialized parameters
class VAE_struc(nn.Module):
    def __init__(self, params):
        super(VAE_struc, self).__init__()

        assert int(params['dimension']) != 2 or int(params['dimension']) != 3, 'Wrong Dimension value, please fill out 2 or 3'
        self.dim = int(params['dimension'])
        self.relu = nn.Mish() if params['act_function'] == 'mish' else nn.ReLU()
        self.mul_ch = True if params['num_channels'] > 1 else False
        
        batches = int(params['batch_size'])
        img_size = int(params['img_size'])
        
        pad_hwd = int((params['kernel_width']-1) / 2)
        kernel_hwd = int(params['kernel_width'])
        kernel_hwd2 = int(params['kernel_p']) 
        conv_input = int(params['num_channels'])
        conv_input2 = int(params['num_filters'])

        stride = int(params['stride_c'])
        stride2 = int(params['stride_p'])
        
        out_channel = int(params['output_classes'])

        if self.dim == 3: # 3d
            self.desired_shape = (batches, conv_input2, img_size//4, img_size//4, img_size//4) # 1 is batch size

            self.conv_layer1 = nn.Sequential(
                nn.Conv3d(in_channels=conv_input, out_channels=conv_input2, kernel_size=kernel_hwd, stride=stride, padding=pad_hwd),
                nn.BatchNorm3d(num_features=conv_input2),
                nn.ReLU(inplace=True),
            )

            self.conv_layer2 = nn.Sequential(
                nn.Conv3d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=kernel_hwd, stride=stride2, padding=pad_hwd),
                nn.BatchNorm3d(num_features=conv_input2),
                nn.ReLU(inplace=True),
            )

            ### reduce calc val (approximation)
            self.pool_layer1 = nn.Sequential(
                nn.AdaptiveAvgPool3d((img_size//4)),
                nn.Dropout(0.2), # non-lenear(activation function) not needed
            )

            ### restore calc val
            self.unpool_layer1 = nn.Upsample(scale_factor=2, mode='nearest')

            self.deconv_layer1 = nn.Sequential(
                nn.ConvTranspose3d(in_channels=conv_input2, out_channels=conv_input2, kernel_size=kernel_hwd, stride=stride, padding=pad_hwd),
                nn.BatchNorm3d(num_features=conv_input2),
                nn.ReLU(inplace=True),
            )

            self.deconv_layer2 = nn.Sequential(
                nn.ConvTranspose3d(in_channels=conv_input2, out_channels=conv_input, kernel_size=kernel_hwd2, stride=stride2),
                # nn.ConvTranspose3d(in_channels=conv_input2, out_channels=out_channel, kernel_size=kernel_hwd2, stride=stride2), # segmentation not looks good
                nn.BatchNorm3d(num_features=conv_input), # fix as out_channels
                nn.ReLU(inplace=True),
            )

        else: # 2d, value 1 wil be replaced to batch_size
            self.desired_shape = (batches, conv_input, img_size, img_size)
            
            ### add self.conv_layer1 = nn.Sequential() # reduce dimension as (B, 4, imgsize//4, imgsize//4)
            ### add self.deconv_layer1 = nn.Sequential() # restore dimension as raw shape
            
            self.fc_layer1 = nn.Sequential(
                nn.Linear(batches*conv_input*(img_size**2), batches*(img_size**2)),
                nn.BatchNorm1d(num_features=batches*(img_size**2)),
                nn.ReLU(inplace=True),
            )

            self.fc_layer2 = nn.Sequential(
                nn.Linear(batches*(img_size**2), batches*((img_size//2)**2)),
                nn.BatchNorm1d(num_features=((img_size//2)**2)),
                nn.ReLU(inplace=True),
            )

            self.fc_layer3 = nn.Sequential(
                nn.Linear(batches*((img_size//2)**2), batches*((img_size//2)**2)),
                nn.BatchNorm1d(num_features=(img_size//2)**2),
                nn.ReLU(inplace=True),
            )

            self.fc_layer4 = nn.Sequential(
                nn.Linear(batches*((img_size//2)**2), batches*(img_size**2)),
                nn.BatchNorm1d(num_features=batches*(img_size**2)),
                nn.ReLU(inplace=True),
            )

            ### probably this layer will be replaced if conv & deconv layer implemented & applied
            ### if multi-class estimation(multi channel = True)
            self.fc_layer5 = nn.Sequential(
                nn.Linear(batches*(img_size**2), batches*conv_input*(img_size**2)),
                nn.BatchNorm1d(num_features=batches*conv_input*(img_size**2)),
                nn.ReLU(inplace=True),
            )


### VAE encoder block
class VAE_enc(VAE_struc):
    def __init__(self, params):
        super(VAE_enc, self).__init__(params)

    def forward(self, x):
        if self.dim == 3: #3D
            x1 = self.conv_layer1(x)
            x2 = self.conv_layer2(x1)

            # for 3D > convert array shape to 1D
            x3 = self.pool_layer1(x2) ### reduce dimension before flatten
            x_out = x3.view(1, -1)

        else: # 2D            
            x1 = self.fc_layer1(x.view(1, -1))
            x_out = self.fc_layer2(x1)

        mu, sig = x_out, x_out

        std = torch.exp(sig/2)
        eps = torch.randn_like(std) # same size as iterable_tensor(std), initialized array(tensor) with Gaussian Distribution values
        z = mu + (std * eps) # latent vector


        return z, mu, sig


### VAE decoder block
class VAE_dec(VAE_struc):
    def __init__(self, params):
        super(VAE_dec, self).__init__(params)

    def forward(self, x):
        if self.dim == 3:
            x1 = x.view(*self.desired_shape)
            x2 = self.unpool_layer1(x1)
            x3 = self.deconv_layer1(x2) 
            x_out = self.deconv_layer2(x3)

        else: # 2d
            x1 = self.fc_layer3(x)
            x2 = self.fc_layer4(x1)
            x3 = self.fc_layer5(x2) if self.mul_ch == True else x2
            x_out = x3.view(*self.desired_shape)

        return x_out 


class VAE_3D(nn.Module):
    def __init__(self, params):
        super(VAE_3D, self).__init__()

        ### ENCODER & DECODER 
        self.enc = VAE_enc(params)
        self.dec = VAE_dec(params)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x_z, mu, sig = self.enc.forward(x)
        x_out = self.dec.forward(x_z)
        x_out = self.output(x_out)

        return x_out, mu, sig   # loss = reconstruction loss + kl divergence
