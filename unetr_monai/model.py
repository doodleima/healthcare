import torch
from models import unet, unetR
# from monai.networks.nets import SwinUNETR#, UNETR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### load UNet for 3d images
def unetModel():
    model = unet.UNet(
             in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).to(device)#, dtype=torch.half)

    return model


def UNETR_MONAI():
    model = unetR.UNETR( # unetr
    # model = unetR_monai_custom.UNETR( # unetr
        in_channels=1, # number of image modalities
        out_channels=5, # number of classes
        img_size=(256, 256, 176),#(192, 192, 192),
        feature_size=12,
        hidden_size=384,#576,#768,
        mlp_dim=3072,
        num_heads=6,#12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.1,
    ).to(device)

    return model


# def SwinUNETR_MONAI():
#     model = SwinUNETR(
#         in_channels=1,
#         out_channels=5,
#         img_size=(256, 256, 192),
#         feature_size=12,
#         num_heads=6,
#         norm_name="instance",
#         drop_rate=0.1,
#         attn_drop_rate=0.1,
#         use_checkpoint=True,
#     ).to(device)

#     return model