from typing import Optional
from itertools import product

from einops import reduce
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTMAEForPreTraining, ViTMAEModel, ViTConfig
from transformers import ViTPreTrainedModel, ViTModel

from .latent_transformers import ImageLatentTransformerGroup
from ..utils.img import horizontal_flip_img, rotate_img, hflip_rotate_img


# default vit config
default_vit_cfg = ViTConfig()


class TI_ViT(nn.Module):
    def __init__(
        self,
        pretrained_dir: Optional[str]=None,
    ):
        """TI_ViT

        Args:
            pretrained_dir (str): Path to the pretraining model. \
                Defaults to "./models/facebook/vit-mae-base".
        """
        super(TI_ViT, self).__init__()
        self.pretrained_dir = pretrained_dir
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        ])

        # load ViTMAE from  checkpoint, ignore decoder, follow PeCLR
        if pretrained_dir is not None:
            self.backbone = ViTModel.from_pretrained(self.pretrained_dir)
        else:
            self.backbone = ViTModel(default_vit_cfg)

        # latent transformation, default config
        self.trans_grp = ImageLatentTransformerGroup()

    def forward_loss(self,
        images: torch.Tensor,
        compute_secondary: bool=False,
    ) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Images within [0,1], size=(N,3,H,W)
            compute_secondary (bool): Toggle secondary loss computation

        Returns:
            torch.Tensor: loss
        """
        batch_size = images.shape[0]
        dtype, device = images.dtype, images.device

        # origin patches
        images_norm = self.image_preprocessor(images)
        patches_origin = self.backbone(images_norm).last_hidden_state[:, 1:]

        # Ordinary Loss
        a = torch.rand(size=(batch_size,), dtype=dtype, device=device) * torch.pi * 2
        b = torch.rand(size=(batch_size,), dtype=dtype, device=device) * torch.pi * 2
        # generate ordinary transformed images
        images_hf = horizontal_flip_img(images_norm)
        images_cr = rotate_img(images_norm, a)
        images_hr = hflip_rotate_img(images_norm, b)
        delta_hf = self.backbone(images_hf).last_hidden_state[:, 1:] - \
            self.trans_grp.do_hf(patches_origin)
        delta_cr = self.backbone(images_cr).last_hidden_state[:, 1:] - \
            self.trans_grp.do_cr(patches_origin, a)
        delta_hr = self.backbone(images_hr).last_hidden_state[:, 1:] - \
            self.trans_grp.do_hr(patches_origin, b)
        loss_ordinary = \
            reduce(delta_hf.abs(), "b l d -> b", reduction="mean").mean() + \
            reduce(delta_cr.abs(), "b l d -> b", reduction="mean").mean() + \
            reduce(delta_hr.abs(), "b l d -> b", reduction="mean").mean()

        # Secondary Loss
        loss_secondary = torch.tensor(0, dtype=dtype, device=device)
        if compute_secondary:
            a1 = torch.rand(size=(batch_size,), dtype=dtype, device=device) * torch.pi * 2
            a2 = torch.rand(size=(batch_size,), dtype=dtype, device=device) * torch.pi * 2
            b1 = torch.rand(size=(batch_size,), dtype=dtype, device=device) * torch.pi * 2
            b2 = torch.rand(size=(batch_size,), dtype=dtype, device=device) * torch.pi * 2

            for ((op1, t_op1, p1), (op2, t_op2, p2)) in product(
                zip([horizontal_flip_img, rotate_img, hflip_rotate_img],
                    [self.trans_grp.do_hf, self.trans_grp.do_cr, self.trans_grp.do_hr],
                    [None, a1, a2]),
                zip([horizontal_flip_img, rotate_img, hflip_rotate_img],
                    [self.trans_grp.do_hf, self.trans_grp.do_cr, self.trans_grp.do_hr],
                    [None, b1, b2]),
            ):
                images_trans = op2(op1(images_norm, p1), p2)
                delta_trans = self.backbone(images_trans).last_hidden_state[:, 1:] - \
                    t_op2(t_op1(patches_origin, p1), p2)
                loss_secondary += reduce(delta_trans.abs(), "b l d -> b", reduction="mean").mean()

        loss = loss_ordinary + 1e-3 * loss_secondary
        return loss