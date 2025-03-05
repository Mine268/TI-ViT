import cv2
import numpy as np
from torchvision import transforms

from sl_vit2.dataset import Ego4DHandImage


def test_Ego4dHandImage():
    dataset = Ego4DHandImage(
        root="/mnt/qnap/data/datasets/ego4d_hand_sep60",
        img_size=224,
        default_augment=True,
        custom_transform=transforms.Compose(
            [
                transforms.RandomRotation(degrees=30)
            ]
        )
    )
    print(f"ego4d dataset loaded, len={len(dataset)}")

    sample = dataset[3293]
    print(sample.shape)

    cv2.imwrite(
        "tests/test_ego4d/sample.png",
        (sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )


if __name__ == "__main__":
    test_Ego4dHandImage()