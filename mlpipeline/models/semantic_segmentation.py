import types

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import monai
from torch import nn
from torchvision import transforms as T
from segmentation_models_pytorch.encoders import encoders
from monai.networks.nets import BasicUnet, BasicUnetPlusPlus
from monai.networks import nets as monai_nets

from mlpipeline.losses import create_segmentation_loss
from mlpipeline.models.segmentation.eoformer.eoformer import EoFormer
from mlpipeline.models.segmentation.swin_umamba.UMambaEncDepthConvK1_3d import get_umamba_enc_dc_k1_3d_from_plans


class SemanticSegmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.create_criterion()
        self.model = self.get_model()

    def get_model(self):
        if self.cfg.arch.lower() == "EoFormer".lower():
            self.model = EoFormer(
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                drop_path=self.cfg.dropout,
            )
        elif self.cfg.arch.lower() == 'Unet3D'.lower():
            self.model = BasicUnet(
                spatial_dims=3,
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                dropout=self.cfg.dropout,
            )
        elif self.cfg.arch.lower() == 'UnetPP3D'.lower():
            self.model = BasicUnetPlusPlus(
                spatial_dims=3,
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                dropout=self.cfg.dropout,
            )
        elif self.cfg.arch.lower() == 'SegResNet'.lower():
            self.model = monai_nets.segresnet.SegResNet(
                spatial_dims=3,
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                dropout_prob=self.cfg.dropout,
            )
        elif self.cfg.arch.lower() == 'UNETR'.lower():
            self.model = monai_nets.unetr.UNETR(
                img_size=(self.cfg.image_size, self.cfg.image_size, self.cfg.image_depth),
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                dropout_rate=self.cfg.dropout,
            )
        elif self.cfg.arch.lower() == 'SwinUNETR'.lower():
            self.model = monai_nets.swin_unetr.SwinUNETR(
                img_size=(self.cfg.image_size, self.cfg.image_size, self.cfg.image_depth),
                feature_size=48,
                in_channels=self.cfg.num_channels,
                out_channels=self.cfg.num_classes,
                drop_rate=self.cfg.dropout,
            )
        elif self.cfg.arch.lower() == "UMambaEncDC_K1".lower():
            self.model = get_umamba_enc_dc_k1_3d_from_plans(
                num_input_channels=self.cfg.num_channels,
                num_output_channels=self.cfg.num_classes,
                depth_mode=self.cfg.depth_mode,
                expand=self.cfg.expand,
                conv_mode=self.cfg.conv_mode,
            )
        else:
            raise ValueError("Invalid architecture!")

        return self.model

    def get_rank(self):
        return next(self.parameters()).device

    def create_criterion(self):
        from_logits = False if self.cfg.arch in ["SkelCon"] else True
        use_opposite_sign = True if self.cfg.use_opposite_sign is None else self.cfg.use_opposite_sign
        am_constant = 0.0 if self.cfg.am_constant is None else self.cfg.am_constant
        directional_weight = 1.0 if self.cfg.directional_weight is None else self.cfg.directional_weight
        distance_weight = 0.5 if self.cfg.distance_weight is None else self.cfg.distance_weight

        self.criterion = create_segmentation_loss(
            self.cfg.loss_name,
            mode=self.cfg.loss_mode,
            from_logits=from_logits,
            smooth=0,
            loss_type=self.cfg.loss_type,
            base_loss=self.cfg.base_loss,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            pos_weight=self.cfg.pos_weight,
            use_opposite_sign=use_opposite_sign,
            am_constant=am_constant,
            directional_weight=directional_weight,
            distance_weight=distance_weight,
        )


    def forward(self, batch):
        label_name = "dist" if ("dist" in self.cfg.label_name) else "gt"
        img_input = batch["input"].cuda(self.get_rank(), non_blocking=True)
        img_gt = batch[label_name].cuda(self.get_rank(), non_blocking=True)

        # print(img_input.amin(dim=(0, 2, 3)), img_input.amax(dim=(0, 2, 3)), img_gt.min(), img_gt.max())
        pred = self.model(img_input)
        assert img_input.ndim >= 4
        assert not hasattr(self.cfg.data, "num_dims") or img_input.ndim >= self.cfg.data.num_dims + 1
        assert not hasattr(self.cfg.data, "num_dims") or img_gt.ndim == self.cfg.data.num_dims + 1

        if isinstance(pred, list) or isinstance(pred, tuple):
            loss = sum([self.criterion(out, img_gt) for out in pred])
            outputs = {"pred": pred[-1], "gt": img_gt}
        else:
            loss = self.criterion(pred, img_gt)
            outputs = {"pred": pred, "gt": img_gt}

        return {"loss": loss}, outputs

    def predict(self, batch, inferer=None):
        label_name = "dist" if ("dist" in self.cfg.label_name) else "gt"
        img_input = batch["input"].cuda(self.get_rank(), non_blocking=True)
        img_gt = batch[label_name].cuda(self.get_rank(), non_blocking=True)

        if self.cfg.test_on_patches and (inferer is not None):
            # SlidingWindowInferer will forcefully get the first output in tuple
            pred = inferer(inputs=img_input, network=self.model)
        else:
            pred = self.model(img_input)

        if isinstance(pred, list) or isinstance(pred, tuple):
            # IterNet, SCSNet
            pred = pred[-1]

        outputs = {"pred": pred, "gt": img_gt}
        return outputs
