from dataclasses import dataclass, asdict
from typing import *
import json


@dataclass
class Config:
    # members
    IMG_SIZE: int = 224

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

    def __post_init__(self):
        if isinstance(self.CAMERA_DEVIATION_RANGE, tuple) and len(self.CAMERA_DEVIATION_RANGE) == 2:
            pass
        elif isinstance(self.CAMERA_DEVIATION_RANGE, list) and len(self.CAMERA_DEVIATION_RANGE) == 2:
            self.CAMERA_DEVIATION_RANGE = tuple(self.CAMERA_DEVIATION_RANGE)
        else:
            raise ValueError("Cannot init CAMERA_DEVICATION_RANGE as [low,high], found "
                f"{self.CAMERA_DEVIATION_RANGE}")


default_cfg = Config()