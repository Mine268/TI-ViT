from sl_vit2.dataset import InterHand26M
from sl_vit2.config import default_cfg
from sl_vit2.utils.misc import breif_dict


def test_InterHand26M():
    dataset = InterHand26M(
        root=default_cfg.ih26m_root,
        transform=lambda x: x,
        data_split="test"
    )

    print(f"length={len(dataset)}")

    inputs, targets, meta_info = dataset[100]
    print("--- inputs")
    breif_dict(inputs)
    print("--- targets")
    breif_dict(targets)
    print("--- meta_info")
    breif_dict(meta_info)
