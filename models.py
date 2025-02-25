import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

class EEGEncoder(nn.Module):
    def __init__(self, in_channels: int=20, img_size: int=64):
        """
        Args:
            in_channels (int): input channels (temporal EEG dimension)
            img_size (int): fMRI spatial size ([img_size, img_size])
        """
        super().__init__()
        self.img_size = img_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 3)), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 3)), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Input: [20, C, 249] (C: electrode channels)
        x = self.conv1(x)   # [32, C, 126]
        x = self.conv2(x)   # [64, C, 64]
        x = self.conv3(x)   # [128, C, 64]
        x = self.conv4(x)   # [256, C, 64]

        _, _, H, W = x.shape
        # resize to the same spatial size as fMRI # [256, W, H]
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bicubic')

        return x

class fMRIDecoder(nn.Module):
    def __init__(self, in_channels: int=256, out_channels: int=30):
        """
        Args:
            out_channels (int): fMRI output channels
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Channels from 32 -> out_channels
        self.conv_channels = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input: [256, W, H]
        x = self.conv1(x)   # [128, W, H]
        x = self.conv2(x)   # [64, W, H]
        x = self.conv3(x)   # [32, W, H]

        x = self.conv_channels(x)  # fMRI dimension [out_channels, W, H]
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels: int=30):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
        )
    
    def forward(self, x):
        return self.model(x)

def create_unet(in_channels=256, out_channels=256):
    return UNet2DModel(
        in_channels=in_channels,  # the number of input channels
        out_channels=out_channels,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        block_out_channels=(out_channels, out_channels),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D", 
            "DownBlock2D",
        ), 
        up_block_types=(
            "UpBlock2D", 
            "UpBlock2D", 
        ),
    )

class EEG2fMRINet(nn.Module):
    def __init__(self, eeg_encoder, unet_module, fmri_decoder):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.unet_module = unet_module
        self.fmri_decoder = fmri_decoder
    
    def forward(self, eeg):
        enc_states = self.eeg_encoder(eeg)
        
        # Sample zero timestep
        bs = enc_states.shape[0]
        timesteps = torch.zeros((bs,), device=enc_states.device).long()
        enc_states = self.unet_module(sample=enc_states, timestep=timesteps).sample

        out = self.fmri_decoder(enc_states)

        return out