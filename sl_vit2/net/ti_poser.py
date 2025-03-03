from typing import *

import torch
import torch.nn as nn
from torchvision import transforms

from .ti_vit import TI_ViT
from .transformer_module import TransformerBlock


class TI_MANOPoser(nn.Module):
    def __init__(
        self,
        pretrained_path: str=None,
        vit_config_path: str=None,
        num_joints: int=16,
        num_proj_layer: int=3,
    ):
        """TI_MANOPoser

        Args:
            num_joints (int): Number of joint.
            pretrained_path (Optional[str], optional): Path to pretrained TI_ViT. Defaults to None.
            vit_config_path (str): Path to architecture config file.
        """
        super(TI_MANOPoser, self).__init__()

        if pretrained_path is None or vit_config_path is None:
            raise ValueError("You should provide TI_ViT config and checkpoints.")

        self.pretrained_path = pretrained_path
        self.vit_config_path = vit_config_path
        self.image_preprocessor = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        ])

        # create model
        self.backbone: TI_ViT = TI_ViT(pretrained_dir=None, config_path=self.vit_config_path)
        self.embed_dim: int = self.backbone.embed_dim

        # load checkpoint
        ckpt = torch.load(self.pretrained_path, map_location=torch.device("cpu"))
        self.backbone.load_state_dict(ckpt["model"])

        # Joint projection layer
        self.num_joints = num_joints
        self.num_proj_layer = num_proj_layer
        # query token
        self.query_tokens = nn.Parameter(torch.randn(size=(self.num_joints+1+1, self.embed_dim)))
        # layers
        self.joint_trans = nn.Sequential(
            *[TransformerBlock(dim=self.embed_dim, num_heads=12) for _ in range(num_proj_layer)]
        )
        self.joint_proj = nn.Linear(
            in_features=self.embed_dim, out_features=6*self.num_joints, bias=True
        )  # 6d rep of rotations
        self.shape_proj = nn.Linear(
            in_features=self.embed_dim, out_features=10, bias=True
        )
        self.root_proj = nn.Linear(
            in_features=self.embed_dim, out_features=3, bias=True
        )  # root joint position, in meter

    def forward(self,
        images: torch.Tensor,  # [B,C,H,W]
        cr_angle: Optional[torch.Tensor]=None,
        hr_angle: Optional[torch.Tensor]=None,
    ):
        """
        Args:
            images (torch.Tensor): Tensor image between [0,1], size=(B,C,H,W)
            cr/hr_angle (torch.Tensor): Angles for latent transformations, size=(B,)
        """
        # pose result
        result = {
            "origin": None,
            "cr": None,
            "hr": None,
        }

        # origin patches
        images_norm = self.image_preprocessor(images)
        patches_origin = self.backbone.encode(images_norm)  # [B,L,D]

        # origin pose
        org_joint_tokens = patches_origin[:, :self.num_joints]
        org_shape_token = patches_origin[:, self.num_joints]
        org_root_token = patches_origin[:, self.num_joints+1]
        result["origin"] = {
            "joint": self.joint_proj(org_joint_tokens),
            "shape": self.shape_proj(org_shape_token),
            "root": self.root_proj(org_root_token),
        }

        if cr_angle is not None:
            patches_cr = self.backbone.trans_grp.do_cr(patches_origin, cr_angle)
            cr_joint_tokens = patches_cr[:, :self.num_joints]
            cr_shape_token = patches_cr[:, self.num_joints]
            cr_root_token = patches_cr[:, self.num_joints+1]
            result["cr"] = {
                "joint": self.joint_proj(cr_joint_tokens),
                "shape": self.shape_proj(cr_shape_token),
                "root": self.root_proj(cr_root_token),
            }

        if hr_angle is not None:
            patches_hr = self.backbone.trans_grp.do_hr(patches_origin, hr_angle)
            hr_joint_tokens = patches_hr[:, :self.num_joints]
            hr_shape_token = patches_hr[:, self.num_joints]
            hr_root_token = patches_hr[:, self.num_joints+1]
            result["hr"] = {
                "joint": self.joint_proj(hr_joint_tokens),
                "shape": self.shape_proj(hr_shape_token),
                "root": self.root_proj(hr_root_token),
            }

        return result