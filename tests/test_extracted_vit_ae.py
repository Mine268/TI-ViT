import json

import cv2
import numpy as np
from einops import rearrange
from transformers import ViTModel, ViTMAEConfig, ViTMAEForPreTraining
import torch
from torchvision import transforms

from sl_vit2.net.transformer_module import ViTMAEDecoder_NoMask, ViTModelFromMAE


def test_mae():
    mae = ViTMAEForPreTraining.from_pretrained("models/facebook/vit-mae-base")
    image_preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    img_cv2 = cv2.imread("tests/test_ego4d/sample.png")
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_batch = image_preprocessor(img_cv2_rgb)[None, ...]

    output = mae.vit(img_batch)
    latent = output.last_hidden_state
    ids_restore = output.ids_restore
    mask = output.mask

    decoder_output = mae.decoder(latent, ids_restore)
    recons = mae.unpatchify(decoder_output.logits, (224, 224))[0]
    recons[0] = recons[0] * 0.229 + 0.485
    recons[1] = recons[1] * 0.224 + 0.456
    recons[2] = recons[2] * 0.225 + 0.406

    img = (recons.detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/test_extracted_vit_ae/recon_mae.png", img_cv2)


def test_extracted_model():
    encoder: ViTModel = ViTModel.from_pretrained("models/facebook/converted-vit-base")
    encoder.pooler = torch.nn.Identity()
    with open("models/facebook/vit-mae-base-decoder/config.json", "r") as f:
        config = json.load(f)
    decoder: ViTMAEDecoder_NoMask = ViTMAEDecoder_NoMask(ViTMAEConfig(**config), 196)
    decoder.load_state_dict(torch.load(
        "models/facebook/vit-mae-base-decoder/vit-mae-base-decoder.pth"
    ))

    image_preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    img_cv2 = cv2.imread("tests/test_ego4d/sample.png")
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_batch = image_preprocessor(img_cv2_rgb)[None, ...]

    patches = encoder(img_batch).last_hidden_state  # [N,1+H'*W',D]
    recons = decoder(patches).logits  # [N,H'*W',3*H*W]
    img = rearrange(
        recons,
        "n (h w) (p q c) -> n (h p) (w q) c",
        h=14, w=14,
        p=16, q=16, c=3
    )[0]
    img[..., 0] = img[..., 0] * 0.229 + 0.485
    img[..., 1] = img[..., 1] * 0.224 + 0.456
    img[..., 2] = img[..., 2] * 0.225 + 0.406

    img = (img.detach().numpy() * 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/test_extracted_vit_ae/recon.png", img_cv2)


def test_ViTModelFromMAE():
    encoder = ViTModelFromMAE.from_pretrained("models/facebook/converted-vit-base")
    with open("models/facebook/vit-mae-base-decoder/config.json", "r") as f:
        config = json.load(f)
    decoder: ViTMAEDecoder_NoMask = ViTMAEDecoder_NoMask(ViTMAEConfig(**config), 196)
    decoder.load_state_dict(torch.load(
        "models/facebook/vit-mae-base-decoder/vit-mae-base-decoder.pth"
    ))

    image_preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    img_cv2 = cv2.imread("tests/test_ego4d/sample.png")
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_batch = image_preprocessor(img_cv2_rgb)[None, ...]

    patches = encoder(img_batch).last_hidden_state  # [N,1+H'*W',D]
    recons = decoder(patches).logits  # [N,H'*W',3*H*W]
    img = rearrange(
        recons,
        "n (h w) (p q c) -> n (h p) (w q) c",
        h=14, w=14,
        p=16, q=16, c=3
    )[0]
    img[..., 0] = img[..., 0] * 0.229 + 0.485
    img[..., 1] = img[..., 1] * 0.224 + 0.456
    img[..., 2] = img[..., 2] * 0.225 + 0.406

    img = (img.detach().numpy() * 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/test_extracted_vit_ae/recon2.png", img_cv2)