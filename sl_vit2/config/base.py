from dataclasses import dataclass, asdict
from typing import *
import json


@dataclass
class Config:
    # members
    # experiment
    exp: str
    data: str
    model_dir: str
    epoch: int = 30

    # data
    COCO_root: str = r"/mnt/qnap/data/datasets/coco2017/train/images"
    ego4d_root: str = r"/mnt/qnap/data/datasets/ego4d_hand_sep60"
    ih26m_root: str = r"/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1"
    img_size: int = 224

    # train
    ft_method: str = "full_param"
    secondary_loss: bool=True
    batch_size: int = 11
    lr: float = 1e-4
    lr_min: float = 1e-6
    optimizer: str = "adamw"
    lr_scheduler: str = "warmup"
    lora_rank: int = 4

    # cosine anneling
    T_0: int = 10
    T_mult: int = 2

    # warmup
    warmup_epoch: int = 1
    cooldown_epoch: int = 10


    # member functions
    def update(self, other: Union['Config', Dict[str, Any]]):
        if isinstance(other, Config):
            merge_dict = other.to_dict()
        elif isinstance(other, dict):
            merge_dict = other
        else:
            raise TypeError("can only merge from Config/dict")

        for key, value in merge_dict.items():
            if hasattr(self, key):
                current_type = type(getattr(self, key))
                if not isinstance(value, current_type):
                    raise ValueError(f"Type mismatched: '{key}' expects {current_type.__name__}, "
                        f"but received {type(value).__name__}.")
                setattr(self, key, value)
            else:
                raise KeyError(f"Unexpected key: {key}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)


default_cfg = Config(
    exp="debug_train",
    data="COCO",
    model_dir="",
)