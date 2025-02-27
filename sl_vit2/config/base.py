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
    epoch: int = 1

    # data
    COCO_root: str = r"/mnt/qnap/data/datasets/coco2017/train/images"
    img_size: int = 224

    # train
    secondary_loss: bool=True
    batch_size: int = 8
    lr: float = 1e-4


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