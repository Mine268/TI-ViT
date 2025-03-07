import argparse
import datetime
from typing import *
import os
import json
from copy import deepcopy

import math
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from sl_vit2.config import *
from sl_vit2.dataset import COCO2017, Ego4DHandImage
from sl_vit2.net import TI_ViT, warmup_scheduler


def nop(*a, **k):
    _, _ = a, k


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(
        backend="nccl",
        rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )


def tensor_item(x: Union[torch.Tensor, float]):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


def get_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def setup(rank: int, cfg: Config, print_func: Callable = print):
    """Setup training models and data"""
    # 0. basic setup
    device = torch.device(f"cuda:{rank}")
    start_epoch = 1
    end_epoch = cfg.epoch

    summary_writer: Optional[SummaryWriter] = None
    if rank == 0:
        summary_writer = SummaryWriter(log_dir=f"./checkpoints/{cfg.exp}/tb_logs")

    # 1. init dataset
    if cfg.data == "COCO":
        dataset = COCO2017(
            cfg.COCO_root,
            cfg.img_size,
        )
    elif cfg.data == "ego4d":
        dataset = Ego4DHandImage(
            cfg.ego4d_root,
            cfg.img_size,
        )
        desired_length = 32560 + 47125  # freihand + youtube3dhands
        dataset, _ = random_split(
            dataset, [desired_length, len(dataset) - desired_length]
        )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=int(os.cpu_count() / get_world_size()),
        sampler=DistributedSampler(dataset),
    )

    # 2. setup model
    model = TI_ViT(cfg.model_dir)
    model.to(rank)
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )

    max_lr = math.sqrt(get_world_size() * cfg.batch_size) * cfg.lr
    min_lr = math.sqrt(get_world_size() * cfg.batch_size) * cfg.lr_min
    optimizer: torch.optim.Optimizer = None
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=max_lr,
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=max_lr,
        )

    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    in_epoch_scheduler: bool = False
    if cfg.lr_scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=min_lr
        )
    elif cfg.lr_scheduler == "warmup":
        scheduler = warmup_scheduler(
            optimizer=optimizer,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_epochs=cfg.warmup_epoch,
            annealing_epochs=cfg.cooldown_epoch,
            steps_per_epoch=len(dataloader),
        )
        in_epoch_scheduler = True

    # loss is computed by model itself
    # 2.1 read the ckpt if found
    if os.path.exists(f"./checkpoints/{cfg.exp}/checkpoint.pt"):
        print_func(
            f"[GPU {rank}] found checkpoints,"
            f" trying reading from ./checkpoints/{cfg.exp}/checkpoint.pt"
        )
        ckpt: Dict[str, Any] = torch.load(
            f"./checkpoints/{cfg.exp}/checkpoint.pt",
            map_location=device,
            weights_only=False,
        )

        start_epoch = ckpt["epoch"] + 1
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    return (
        start_epoch,
        end_epoch,
        summary_writer,
        dataloader,
        optimizer,
        scheduler,
        in_epoch_scheduler,
        model,
    )


def train_one_epoch(
    rank: int,
    cfg: Config,
    epoch: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Tuple[bool, torch.optim.lr_scheduler.LRScheduler],
    log_every: int = 20,
    print_func: Callable = None,
    summary_writer: SummaryWriter = None,
):
    device = model.device
    start_log_time, end_log_time = datetime.datetime.now(), None

    in_step_scheduler, scheduler_ = scheduler

    for it, images in enumerate(dataloader):
        images = images.to(device)

        # forward
        losses: Dict = model(images, compute_secondary=cfg.secondary_loss)

        # backward
        loss = losses["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if in_step_scheduler and scheduler_ is not None:
            scheduler_.step()

        # log
        if log_every is not None and (it + 1) % log_every == 0:
            # log to tensorboard
            if summary_writer is not None:
                for key, value in losses.items():
                    summary_writer.add_scalar(
                        f"train/{key}",
                        tensor_item(value),
                        global_step=epoch * len(dataloader) + it + 1,
                    )
                summary_writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step=epoch * len(dataloader) + it + 1,
                )
            # log to terminal
            end_log_time = datetime.datetime.now()
            time_between_logs = (end_log_time - start_log_time) / log_every

            header = (
                f"[GPU {rank}]     epoch={epoch} progress={it + 1}/{len(dataloader)} "
            )
            body = f"iter_time={time_between_logs} loss={loss.item()}"
            print_func(header + body)

            start_log_time = datetime.datetime.now()

    if not in_step_scheduler and scheduler_ is not None:
        scheduler_.step()


def main(rank: int, cfg: Config, print_func: Callable = print):
    # 1. setup
    device = torch.device(f"cuda:{rank}")
    (
        start_epoch,
        end_epoch,
        summary_writer,
        dataloader,
        optimizer,
        scheduler,
        in_step_scheduler,
        model,
    ) = setup(rank, cfg, print_func)

    # 2. train
    for epoch in range(start_epoch, end_epoch + 1):
        start_time = datetime.datetime.now()
        print_func(
            f"[GPU {rank}] training for epoch {epoch}/{end_epoch},"
            f" start time {start_time.strftime('%Y-%m-%d_%H-%M-%S')}."
        )

        train_one_epoch(
            rank=rank,
            cfg=cfg,
            epoch=epoch,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=(in_step_scheduler, scheduler),
            print_func=print_func,
            summary_writer=summary_writer,
        )

        end_time = datetime.datetime.now()
        print_func(
            f"[GPU {rank}] epoch {epoch} ends at {end_time.strftime('%Y-%m-%d_%H-%M-%S')},"
            f" time costs {end_time - start_time}."
        )

        # dump checkpoints to file
        torch.distributed.barrier()
        if rank == 0:
            print_func(f"[GPU {rank}] writing checkpoint for epoch {epoch}.")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"./checkpoints/{cfg.exp}/checkpoint_{epoch}.pt",
            )
            if os.path.exists(f"./checkpoints/{cfg.exp}/checkpoint.pt"):
                os.remove(f"./checkpoints/{cfg.exp}/checkpoint.pt")
            os.symlink(
                f"./checkpoint_{epoch}.pt", f"./checkpoints/{cfg.exp}/checkpoint.pt"
            )
        torch.distributed.barrier()

        print_func()  # print a blank line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train in ddp")
    parser.add_argument("--exp", type=str, required=True, help="Exp name")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Data for train",
        choices=["COCO", "ego4d"],
    )
    parser.add_argument("--model_dir", type=str, required=False, help="Model ckpt path")
    parser.add_argument(
        "--secondary_loss",
        type=bool,
        required=True,
        help="Toggle secondary loss",
        default=True,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="adamw",
        help="Optimizer for pretrain",
        choices=["adamw", "sgd"],
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        required=False,
        default="warmup",
        help="Learning rate scheduler",
        choices=["warmup", "cosine_annealing", "constant"],
    )
    args = parser.parse_args()
    exp_name: str = args.exp

    cfg = deepcopy(default_cfg)

    ddp_setup()
    torch.manual_seed(42)
    np.random.seed(42)
    rank = get_rank()

    # read from existing json if exists, otherwise udpate default with input args
    if os.path.exists(f"./checkpoints/{exp_name}/config.json"):
        with open(f"./checkpoints/{exp_name}/config.json", "r") as f:
            json_obj = json.loads(f.read())
        cfg = Config(**json_obj)
    else:
        cfg.update(vars(args))
        # save experiement setup to file
        if rank == 0:
            os.makedirs(f"./checkpoints/{exp_name}", exist_ok=True)
            with open(f"./checkpoints/{exp_name}/config.json", "w") as f:
                f.write(cfg.to_json())

    main(rank, cfg, print_func=print if rank == 0 else nop)
    torch.distributed.destroy_process_group()
