import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

from sl_vit2.utils.img import crop_tensor_with_normalized_box


def test_crop_tensor_with_normalized_box():
    img = cv2.imread("tests/test_utils_img/origin_image.png")
    img = ToTensor()(img)
    img_crop = crop_tensor_with_normalized_box(
        img,
        [0.6, 1.0, 0.5, 0.9],
        [120, 120]
    )
    
    img_crop = img_crop.permute(1, 2, 0)
    cv2.imwrite(
        "tests/test_utils_img/cropped.jpg",
        (img_crop.numpy() * 255).astype(np.uint8)
    )
