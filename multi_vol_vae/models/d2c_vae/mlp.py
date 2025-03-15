import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch import distributions as dist

from models.d2c_vae.blocks import ToRGB, StyledResBlock, SinusoidalPosEmb, ResnetBlockFC
from utils.general_utils import normalize_coordinate, singleplane_positional_encoding, triplane_positional_encoding, sample_plane_feature

class MLP(nn.Module):
    def __init__(self, *, in_ch=2, latent_dim = 64, out_ch=3, ch=256):
        super().__init__()
        self.latent_dim = latent_dim
        ## Scale-aware Injetion Layers
        dim = int(ch // 4)
        sinu_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(dim, ch),
                nn.GELU(),
                nn.Linear(ch, ch)
                )
        activation = 'sin' # sinusoidal activation function 
        ## MLP layers (x8)
        self.net_res1 = StyledResBlock(in_ch + latent_dim, ch, 1, ch, demodulate = True, activation = activation)
        self.net_res2 = StyledResBlock(ch + in_ch + latent_dim, ch, 1, ch, demodulate = True, activation = activation)
        self.net_res3 = StyledResBlock(ch + in_ch + latent_dim, ch, 1, ch, demodulate = True, activation = activation)
        self.net_res4 = StyledResBlock(ch, ch, 1, ch, demodulate = True, activation = activation)
        self.net_res5 = StyledResBlock(ch, ch, 1, ch, demodulate = True, activation = activation)
        # self.net_res6 = StyledResBlock(ch, ch, 1, ch, demodulate = True, activation = activation)
        # self.net_res7 = StyledResBlock(ch, ch, 1, ch, demodulate = True, activation = activation)
        self.torgb = ToRGB(ch, out_ch, ch, upsample = False)

    def forward(self, coords, hdbf, si=1):
        # Enables us to compute gradients w.r.t. coordinates
        coords = coords.clone().detach().requires_grad_(True)
        device = coords.device

        _, c, h, w = coords.shape
        b = hdbf[0].shape[0]

        assert hdbf is not None and len(hdbf) == 3 # Now, only supports three decomposed BFs.
        coords = coords.repeat(b, 1, 1, 1) # [bsz, 1, w, h]
        scale_inj_pixel = torch.ones_like(coords) * si # to be concatenated w x, size: [bsz, 1, h, w]
        coords = coords.permute(0, 2, 3, 1).contiguous() # This is to have the size : [bsz, h, w, 2]
        scale_inj = torch.ones((b,), device = device) * si # This is just a vector of ones of dimension the bsz
        style = self.time_mlp(scale_inj) # [4, 256] [bsz, resolution] # The latent, this is modulating the convolutions later on
        ## Coarse
        ## This part is compute the bilinear interpolation of the coordinate points based on the hdbf
        ############################################################################################
        # hdbf[0] -> torch.Size([4, 64, 128, 128]), coords dim: [bsz, h, w, 1]
        x_ = singleplane_positional_encoding(hdbf[0], coords)
        x_ = torch.cat((x_, scale_inj_pixel), dim = 1)
        
        ## Middle
        # hdbf[1] -> torch.Size([4, 64, 256, 256])
        x_m = singleplane_positional_encoding(hdbf[1], coords)
        x_m = torch.cat((x_m, scale_inj_pixel), dim = 1)
        
        ## Fine
        # hdbf[2] -> torch.Size([4, 64, 256, 256])
        x_h = singleplane_positional_encoding(hdbf[2], coords)
        x_h = torch.cat((x_h, scale_inj_pixel), dim = 1)

        x1 = self.net_res1(x_, style)
        x1 = torch.cat((x1, x_m), dim = 1) # Concatenation 
        
        x2 = self.net_res2(x1, style)
        x2 = torch.cat((x2, x_h), dim = 1)
        
        x3 = self.net_res3(x2, style)
        
        x4 = self.net_res4(x3, style)
        out = self.net_res5(x4, style)
        # x6 = self.net_res5(x5, style)        
        # x7 = self.net_res5(x6, style)      
        # out = self.net_res5(x7, style)         
        
        out = self.torgb(out, style)
        
        return out


