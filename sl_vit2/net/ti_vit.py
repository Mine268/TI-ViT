from functools import partial
from typeguard import typechecked
from typing import Optional, List
from itertools import product

import json
import math
from einops import reduce, rearrange
import torch
import torch.nn as nn
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from transformers import ViTConfig, ViTMAEConfig

from .latent_transformers import ImageLatentTransformerGroup
from .transformer_module import ViTModelFromMAE, ViTMAEDecoder_NoMask
from ..utils.img import horizontal_flip_img, rotate_img, hflip_rotate_img


class SupportLoss(nn.Module):
    def __init__(self, support: float, alpha: float=1e-3):
        super().__init__()
        self.support = support
        self.alpha = alpha
        self.inv_support = 1.0 / support

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_norms = torch.norm(tokens, p=2, dim=-1)
        mean_norm = torch.mean(token_norms)

        delta = self.support - mean_norm

        if delta > -1e-6:
            return self.alpha * delta ** 2
        else:
            return -delta * torch.log(mean_norm * self.inv_support)


# default vit config
default_vit_cfg = ViTConfig()


class TI_ViT(nn.Module):
    @classmethod
    @typechecked
    def setup_lora_model(
        cls,
        model: "TI_ViT",
        backbone_target_modules: List = ["query", "key", "value"],
        backbone_lora_rank: int = 1,
        decoder_target_modules: Optional[List] = None,
        decoder_lora_rank: Optional[int] = None,
    ) -> "TI_ViT":
        backbone_lora_config = LoraConfig(
            r=backbone_lora_rank,
            lora_alpha=32,
            target_modules=backbone_target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[]
        )
        model.backbone = get_peft_model(model.backbone, backbone_lora_config)
        if decoder_target_modules is not None and decoder_lora_rank is not None:
            decoder_lora_config = LoraConfig(
                r=decoder_lora_rank,
                lora_alpha=32,
                target_modules=decoder_target_modules,
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[]
            )
            model.decoder = get_peft_model(model.decoder, decoder_lora_config)
        else:
            model.decoder.eval()
        return model

    def __init__(
        self,
        backbone_ckpt_dir: Optional[str]=None,
        backbone_arch_path: Optional[str]=None,
        decoder_ckpt_path: Optional[str]=None,
        decoder_arch_path: Optional[str]=None,
    ):
        """TI_ViT

        Args:
            backbone_ckpt_dir (str): Path to the pretraining model. \
                Defaults to "./models/facebook/vit-mae-base".
            backbone_arch_path (str): Path to architecture config json file.
            decoder_arch_path: (str): Path to decoder architecture json file. Leaving `None` will \
                ignore the decoder and reconstruction loss during pretraining.
            decoder_ckpt_path (str): Path to decoder checkpoint file.
        """
        super(TI_ViT, self).__init__()
        self.backbone_ckpt_dir = backbone_ckpt_dir
        self.backbone_arch_path = backbone_arch_path
        self.decoder_ckpt_path = decoder_ckpt_path
        self.decoder_arch_path = decoder_arch_path
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        ])

        # --- Backbone part ---
        # load ViTMAE from  checkpoint, ignore decoder, follow PeCLR
        if backbone_ckpt_dir is not None:
            self.backbone = ViTModelFromMAE.from_pretrained(self.backbone_ckpt_dir)
        else:
            with open(self.backbone_arch_path, "r") as f:
                backbone_config = json.load(f)
            self.backbone = ViTModelFromMAE(ViTConfig(**backbone_config))

        # hidden size
        self.embed_dim: int = self.backbone.config.hidden_size
        self.img_size: int = self.backbone.config.image_size
        self.patch_size: int = self.backbone.config.patch_size
        self.num_p: int = self.img_size // self.patch_size
        self.num_patches: int = self.num_p ** 2

        # --- Decoder part ---
        self.decoder: nn.Module = nn.Identity()
        self.enable_decoder: bool = False
        if decoder_arch_path is not None:
            if self.decoder_arch_path is not None:
                with open(self.decoder_arch_path, "r") as f:
                    decoder_config = json.load(f)
            else:
                decoder_config = {}
            self.decoder = ViTMAEDecoder_NoMask(
                ViTMAEConfig(**decoder_config),
                self.num_patches
            )
            self.decoder.load_state_dict(torch.load(self.decoder_ckpt_path))
            self.enable_decoder = True

        # --- Latent transformation part ---
        self.trans_grp = ImageLatentTransformerGroup()

        # support loss
        self.support_distant: float = math.sqrt(self.embed_dim)
        self.support_loss = SupportLoss(self.support_distant)

    def forward(self,
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
        tokens = self.backbone(images_norm).last_hidden_state
        patches_origin = tokens[:, 1:]

        # --- Reconstruction Loss ---
        loss_recons: torch.Tensor = torch.tensor(0, dtype=dtype, device=device)
        if self.enable_decoder:
            images_recons = self.decoder(tokens).logits
            images_norm_patches = rearrange(
                images_norm,
                "n c (h p) (w q) -> n (h w) (p q c)",
                c=3,
                h=self.num_p, w=self.num_p,
                p=self.patch_size, q=self.patch_size
            )
            loss_recons = (images_recons - images_norm_patches).abs().mean(-1).mean()

        # --- Ordinary Loss ---
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
        loss_ordinary: torch.Tensor = \
            reduce(delta_hf.abs(), "b l d -> b", reduction="mean").mean() + \
            reduce(delta_cr.abs(), "b l d -> b", reduction="mean").mean() + \
            reduce(delta_hr.abs(), "b l d -> b", reduction="mean").mean()
        loss_support: torch.Tensor = self.support_loss(
            torch.cat([
                delta_hf,
                delta_cr,
                delta_hr
            ], dim=0)
        )

        # --- Secondary Loss ---
        loss_secondary: torch.Tensor = torch.tensor(0, dtype=dtype, device=device)
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
                composed_op = self.trans_grp.compose(
                    partial(t_op1, angle_rad=p1),
                    partial(t_op2, angle_rad=p2)
                )
                delta_trans = self.backbone(images_trans).last_hidden_state[:, 1:] - \
                    composed_op(patches_origin)
                loss_secondary += reduce(delta_trans.abs(), "b l d -> b", reduction="mean").mean()

        loss = (loss_ordinary + 1e-1 * loss_support) + 1e-3 * loss_secondary + 1e-3 * loss_recons
        return {
            "loss": loss,
            "ordinary": loss_ordinary.item(),
            "support": loss_support.item(),
            "secondary": loss_secondary.item(),
            "recons": loss_recons.item(),
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode the images into patches.

        Args:
            images (torch.Tensor): Tensor image between [0,1]. size=(B,C,H,W), where H=W=224.

        Returns:
            torch.Tensor: Patches, size=(B,(H//P * W//P),D).
        """
        images_norm = self.image_preprocessor(images)
        patches = self.backbone(images_norm).last_hidden_state[:, 1:]
        return patches