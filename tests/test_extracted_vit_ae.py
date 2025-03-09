import json

import cv2
import numpy as np
from einops import rearrange
from transformers import ViTModel, ViTMAEConfig, ViTMAEForPreTraining
import torch
from torchvision import transforms

from sl_vit2.net.transformer_module import ViTMAEDecoder_NoMask


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

    def unpatchify(patchified_pixel_values, patch_size, num_channels, original_image_size):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`Tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = patch_size, num_channels
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

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
    img = unpatchify(recons, 16, 3, (224, 224))[0].permute(1, 2, 0)
    img[..., 0] = img[..., 0] * 0.229 + 0.485
    img[..., 1] = img[..., 1] * 0.224 + 0.456
    img[..., 2] = img[..., 2] * 0.225 + 0.406

    img = (img.detach().numpy() * 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/test_extracted_vit_ae/recon.png", img_cv2)