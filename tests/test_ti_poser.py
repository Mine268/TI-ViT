import torch
import numpy as np

from sl_vit2.net import TI_MANOPoser
from sl_vit2.utils.misc import breif_dict


def test_ti_poser_load():
    ti_poser = TI_MANOPoser("checkpoints/debug_train/checkpoint_1.pt",
        "models/facebook/converted-vit-base/config.json").to("cuda")


def test_ti_poser():
    print()

    ti_poser = TI_MANOPoser(pretrained_path="checkpoints/debug_train/checkpoint_1.pt",
                            vit_config_path="models/facebook/converted-vit-base/config.json").to("cuda:0")
    images = torch.randn(size=(4,3,224,224)).to("cuda:0")

    y1 = ti_poser(images)
    breif_dict(y1)

    cr = torch.randn(size=(4,)).to("cuda:0")
    hr = torch.randn(size=(4,)).to("cuda:0")

    print("---")
    y2 = ti_poser(images, cr)
    breif_dict(y2)

    print("---")
    y3 = ti_poser(images, None, hr)
    breif_dict(y3)

    print("---")
    y4 = ti_poser(images, cr, hr)
    breif_dict(y4)