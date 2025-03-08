import pytest
import pickle as pkl

from sl_vit2.dataset import InterHand26M
from sl_vit2.config import default_cfg
from sl_vit2.utils.misc import breif_dict

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_InterHand26M():
    dataset = InterHand26M(
        root=default_cfg.ih26m_root,
        transform=lambda x: x,
        data_split="test"
    )

    print(f"length={len(dataset)}")

    inputs, targets, meta_info = dataset[1392]
    print("--- inputs")
    breif_dict(inputs)
    print("--- targets")
    breif_dict(targets)
    print("--- meta_info")
    breif_dict(meta_info)

    print("--- sample dumped to tests/test_InterHand26M")
    with open("tests/test_InterHand26M/sample.pkl", "wb") as f:
        pkl.dump({
            "inputs": inputs,
            "targets": targets,
            "meta_info": meta_info,
        }, f)


def test_IH26M_data():
    with open("tests/test_InterHand26M/sample.pkl", "rb") as f:
        sample = pkl.load(f)

    inputs = sample["inputs"]
    targets = sample["targets"]
    meta_info = sample["meta_info"]
    print("--- inputs")
    breif_dict(inputs)
    print("--- targets")
    breif_dict(targets)
    print("--- meta_info")
    breif_dict(meta_info)

