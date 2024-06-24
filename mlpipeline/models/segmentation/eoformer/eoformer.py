import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from mlpipeline.models.segmentation.eoformer.encoder_and_decoder import EHE_encoder
from mlpipeline.models.segmentation.eoformer.encoder_and_decoder import EoFormer_decoder


class EoFormer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path, base_channels=16, spatial_dims=3, norm_name=("instance")):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = out_channels
        self.n_channels = base_channels
        
        self.encoder = EHE_encoder(in_chans=self.in_channels, drop_path_rate=drop_path)
        self.decoder = EoFormer_decoder(drop_path_rate=drop_path)
        
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.n_channels, 
            out_channels=self.n_classes
        )

    def forward(self, x):
        encout_x, intmd_output = self.encoder(x)
        dec_out = self.decoder(intmd_output)
        out = self.out(dec_out)
        return out


if __name__ == "__main__":

    model = EoFormer(in_channels=4, out_channels=3, drop_path=0.1)
    model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total parameters count: {pytorch_total_params:.2f}M.")
    
    start = time.time()
    img = torch.rand([1, 4, 128, 128, 128]).cuda()
    output= model(img)
    end = time.time()
    print(f'time consuming {(end - start)}s.')
    