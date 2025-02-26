from typing import Optional

import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import ViTMAEForPreTraining, ViTMAEModel, ViTConfig


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
            transforms.Resize(size=(224, ), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        ])
        # load ViTMAE from  checkpoint, ignore decoder, follow PeCLR
        if pretrained_dir is not None:
            self.backbone = ViTMAEForPreTraining.from_pretrained(self.pretrained_dir).base_model
        else:
            self.backbone = ViTMAEModel(default_vit_cfg)
