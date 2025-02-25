import torch

from sl_vit2.utils.img import denormalize, save_tensor_img
from sl_vit2.dataset import COCO2017


def test_COCO2017():
    dataset = COCO2017(r"/mnt/qnap/data/datasets/coco2017/train", 256)
    print("dataset length = ", len(dataset))
    sample = dataset[203]
    print(sample.shape)

    mean = std = torch.tensor([0.5, 0.5, 0.5])
    origin_image = denormalize(sample, mean, std)
    save_tensor_img(origin_image, "tests/test_COCO2017/origin_image.png")
    print("image saved to tests/test_COCO2017/origin_image.png")