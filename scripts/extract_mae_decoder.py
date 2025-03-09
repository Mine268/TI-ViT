from typing import *

import os
from transformers import ViTMAEForPreTraining
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder
import torch


def extract_mae_to_decoder(
    mae_model_path: str,
    output_path: str,
):
    # load the model
    mae_model: ViTMAEForPreTraining = ViTMAEForPreTraining.from_pretrained(mae_model_path)
    decoder: ViTMAEDecoder = mae_model.decoder
    decoder.config.to_json_file(os.path.join(output_path, "config.json"))
    torch.save(decoder.state_dict(), os.path.join(output_path, "vit-mae-base-decoder.pth"))


if __name__ == "__main__":
    extract_mae_to_decoder(
        "/data_1/renkaiwen/TI-ViT/models/facebook/vit-mae-base",
        "/data_1/renkaiwen/TI-ViT/models/facebook/vit-mae-base-decoder"
    )