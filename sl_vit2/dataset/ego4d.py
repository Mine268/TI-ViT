from typing import *

import json
from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset


class Ego4DHandImage(Dataset):
    """Hand image dataset from Ego4D, automatically annotated by mediapipe
    """
    def __init__(
        self,
        root: str,
    ):
        super(Ego4DHandImage, self).__init__()
        self.root = Path(root)
        self.image_root = self.root / "images"
        self.annot_root = self.root / "annotations"

        # read annotations
        self.annotations = []
        for annot_file in self.annot_root.iterdir():
            if annot_file.suffix != ".json":
                continue
            with open(annot_file.as_posix(), "r") as f:
                video_annot: Dict[str, Dict[str, Any]] = json.load(f)
            for _, frame_annot in video_annot.items():
                for bbox in frame_annot["hands"].values():
                    self.annotations.append({
                        "frame_path": frame_annot["image_path"],
                        "bbox": [
                            bbox["bbox"]["x_min"],
                            bbox["bbox"]["x_max"],
                            bbox["bbox"]["y_min"],
                            bbox["bbox"]["y_max"],
                        ]
                    })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, ix: int) -> torch.Tensor:
        return