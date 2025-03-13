import cv2
import numpy as np

from sl_vit2.dataset import HIntHandImage


def test_HInt_overall():
    dataset = HIntHandImage(
        root="/mnt/qnap/data/datasets/HInt/HInt_annotation_partial/",
        img_size=224,
        parts=["epick", "newdays"],
        default_augment=False,
        custom_transform=None
    )

    print(f"length of dataset={len(dataset)}")

    image = dataset[234]
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./tests/test_HInt/image.png", image)