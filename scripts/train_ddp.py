import os
from argparse import Namespace
from time import time

import torch
from torch.utils.data import DataLoader

import sys
# sys.path.append('/home/robin/fareal/icassp_grasp/')
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

from lib.datasets import create_dataset
from lib.datasets.grasp_data import grasp_data_collate
from lib.models.model_abc import ModelABC as ModelABC
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix, format_args_cfg
from lib.utils.net_utils import (build_optimizer, build_scheduler, clip_gradient, setup_seed, worker_init_fn)
from lib.utils.recorder import Recorder
from lib.utils.summarizer import SummaryWriter  # 改为普通SummaryWriter


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=0, time_f=time_f, root_path="./checkpoints")
    summary = SummaryWriter(log_dir=recorder.tensorboard_path)

    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    train_loader = DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=True,
                              num_workers=int(arg.workers),
                              pin_memory=True,
                              drop_last=True,
                              worker_init_fn=worker_init_fn,
                              collate_fn=grasp_data_collate,
                              persistent_workers=(int(arg.workers) > 0))

    val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    val_loader = DataLoader(val_data,
                            batch_size=arg.val_batch_size,
                            shuffle=False,
                            num_workers=int(arg.workers),
                            drop_last=False,
                            worker_init_fn=worker_init_fn,
                            collate_fn=grasp_data_collate)

    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)
    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    device = torch.device(f"cuda:{arg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    transform.to(device)

    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)
    epoch = 0
    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume, arg.resume_epoch)

    n_steps = len(train_loader) * cfg.TRAIN.EPOCH
    logger.warning(f"===== start training [{epoch}, {cfg.TRAIN.EPOCH}), total iters: {n_steps} >>>>")
    
    for epoch_idx in range(epoch, cfg.TRAIN.EPOCH):
        model.train()
        trainbar = etqdm(train_loader, rank=0)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = transform(batch)
            prd, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
            loss = loss_dict["loss"]
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()

            trainbar.set_description(f"{bar_prefix['train']} Epoch {epoch_idx} | {loss.item():.4f}")

        scheduler.step()
        logger.info(f"{bar_prefix['train']} Epoch {epoch_idx} | loss: {loss.item():.4f}, Done")
        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        model.on_train_finished(recorder, epoch_idx)

        if (epoch_idx != cfg.TRAIN.EPOCH - 1 and epoch_idx % arg.eval_freq == 0):
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=0)
                for bidx, batch in enumerate(valbar):
                    step_idx = epoch_idx * len(val_loader) + bidx
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch = transform(batch)
                    prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

            model.on_val_finished(recorder, epoch_idx)

    # do last evaluation
    logger.info("do last validation and save results")
    with torch.no_grad():
        model.eval()
        valbar = etqdm(val_loader, rank=0)
        for bidx, batch in enumerate(valbar):
            step_idx = epoch_idx * len(val_loader) + bidx
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = transform(batch)
            prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

    model.on_val_finished(recorder, epoch_idx)


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    
    logger.info(f"Using single GPU: {arg.gpu_id}")
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Use Single GPU Training <====")
    logger.info(f"Loading config from {arg.cfg}")
    
    main(cfg, arg, exp_time)