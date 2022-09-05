import torch
from models import unet, unetR, unetR_monai

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


### load UNetR for 3d images
def unetRModel():
    model = unetR.UNetR( 
        img_shape=(128, 128, 128), 
        input_dim=4, output_dim=3, # input_dim = number of modalities (T1w, Flair..)
        embed_dim=768, 
        patch_size=16, 
        num_heads=12, 
        dropout=0.1
    ).to(device)#, dtype=torch.half)

    return model


def UNETR_MONAI():
    model = unetR_monai.UNETR( # unetr
        in_channels=1, # number of image modalities
        out_channels=5, # number of classes
        img_size=(128, 128, 128),
        feature_size=16, # patch size
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.1,
    ).to(device)

    return model
