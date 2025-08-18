import torch
import numpy as np
import xarray as xr
from torch import nn
import einops
from omegaconf import OmegaConf
from src.models.components.resnets import ResnetBlock, ConvResnetBlock
from src.models.components.interpolate import Interpolate

class AnomRecon(nn.Module):
    def __init__(
            self,
            dropout: float = 0.,
            num_indexes: list = [7,],
            n_months: int = 12,
            num_pca: int = 0,
            flin_hidden = 32,
            fconv_in = 64,
            fconv_out = 16,
            ) -> None:
        super(AnomRecon, self).__init__()
        self.dropout = dropout
        self.num_indexes = num_indexes
        self.n_months = n_months
        self.num_pca = num_pca
        self.flin_hidden = flin_hidden
        self.fconv_in = fconv_in
        self.fconv_out = fconv_out
        
        # custom feature number
        flin_in = 1 + 12 + np.sum(self.num_indexes)
        flin_out  =  self.fconv_in - 4
        assert(flin_out >= flin_hidden)
        
        # model: linear resnet layers
        self.linear = nn.Sequential(
            nn.Linear(flin_in, self.flin_hidden),
            nn.ReLU(),
            ResnetBlock(self.flin_hidden, self.flin_hidden),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            ResnetBlock(self.flin_hidden, flin_out),
        )
        
        # model: convolutional resnet layers
        inner_convs_layers = []
        for i in reversed(range(
                int(np.log2(self.fconv_out) + 1),
                int(np.log2(self.fconv_in) + 1)
            )):  
            inner_convs_layers.append(ConvResnetBlock(2**i, 2**(i-1)))
            inner_convs_layers.append(nn.ReLU())

        self.conv = nn.Sequential(
            ConvResnetBlock(self.fconv_in, self.fconv_in),
            nn.ReLU(),
            *inner_convs_layers,
            ConvResnetBlock(self.fconv_out, 1),
        )

    def forward(self, input):
        year, month, index, static_data = input

        # concatenate date with indexes
        month_embed = nn.functional.one_hot((month-1).long(), num_classes=12)
        x = torch.cat((year.unsqueeze(1), month_embed, index), dim=1)

        # linear layer
        x = self.linear(x)

        # replicate the features to match the spatial dimensions of the output (181x360)
        h, w = static_data.shape[-2:]  # height, width of the image
        x = einops.repeat(x, 'b f -> b f h w', h=h, w=w)  # b = batch size, f = features

        # concatenate with the static data
        x = torch.cat((x, static_data), dim=1)
        
        # convolutional layer
        x = self.conv(x)

        # remove the (single) channel
        x = x.squeeze(1)
        
        return x
