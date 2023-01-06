import torch
from monai.networks.nets import unet, unetr, DynUNet, SwinUNETR

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
    model = unetr.UNETR( # unetr.UNETR
        in_channels=1, # number of image modalities
        out_channels=5, # number of classes
        img_size=(64, 64, 64), # 176
        feature_size=16,
        hidden_size=1280, # num_heads * 64
        mlp_dim=5120, # hidden_size * heads/4
        num_heads=16,
        pos_embed="conv",
        norm_name="instance", # do not use batch: 1~2,,
        conv_block=True,
        res_block=True,
        dropout_rate=0.2,
        # spatial_dims=3
    ).to(device)

    return model


### nnUNet
# https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/dynunet.py
def NNUNET_MONAI():
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        strides=(1, 2, 2),
        kernel_size=(3, 3, 3),
        upsample_kernel_size=(2,2,2),
        res_block=True,
        dropout=0.1,
    ).to(device)

    return model


### Swin UNetR
### https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py
def SwinUNETR_MONAI():
    model = SwinUNETR( # swinunetr.SwinUNETR
        in_channels=1,
        out_channels=6,
        img_size=(160, 160, 160),
        feature_size=24,
        depths=(1, 1, 2, 2),
        num_heads=(1, 2, 3, 6),
        norm_name="instance", # {'LAYER', 'BATCH', 'LOCALRESPONSE', 'INSTANCE', 'INSTANCE_NVFUSER', 'GROUP', 'SYNCBATCH'}
        # norm_name=("group", {"num_groups":1}), 
        drop_rate=0.2,
        attn_drop_rate=0.2,
        use_checkpoint=True, # reduce memory using ratio
    ).to(device)

    return model

