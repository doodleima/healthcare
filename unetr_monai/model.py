import torch
from models import unetR
from monai.networks.nets import unet, unetr, dynunet # swinunetr, 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Basic UNet
### https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/basic_unet.py
def UNET_MONAI():
    model = unet.UNet( # unet.UNet
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=2,
    ).to(device)

    return model


### Basic UNetR
### https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
def UNETR_MONAI():
    model = unetR.UNETR( # unetr.UNETR
        in_channels=1, # number of image modalities
        out_channels=5, # number of classes
        img_size=(256, 256, 176),
        feature_size=12,
        hidden_size=384, # num_heads * 64
        mlp_dim=3072,
        num_heads=6,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.1,
    ).to(device)

    return model


### Swin UNetR
### https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py
# def SwinUNETR_MONAI():
#     model = swinunetr.SwinUNETR( # swinunetr.SwinUNETR
#         in_channels=1,
#         out_channels=5,
#         img_size=(256, 256, 176),
#         feature_size=12,
#         num_heads=6,
#         norm_name="instance",
#         drop_rate=0.1,
#         attn_drop_rate=0.1,
#         use_checkpoint=True, # reduce memory using ratio
#     ).to(device)

#     return model


### nnUNet
# https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/dynunet.py
def NNUNET_MONAI():
    model = dynunet.DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        strides=(2, 2),
        res_block=True,
        dropout=0.1,
    ).to(device)

    return model