import argparse
import os
from pathlib import Path
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_data import grasp_data_collate
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.viztools.utils import ColorsMap as CMap
from lib.viztools.viz_o3d_utils import VizContext

import torch.nn as nn
from collections import defaultdict


class ActionClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_dims=[64, 256, 64], token_compression_ratios=[4, 4, 4, 64]):
        super().__init__()
        
        self.initial_conv = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=1)
        
        self.channel_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.channel_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
        
        self.token_compressors = nn.ModuleList()
        for ratio in token_compression_ratios:
            self.token_compressors.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=ratio, stride=ratio),
                    nn.BatchNorm1d(hidden_dims[-1]),
                    nn.ReLU()
                )
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial_conv(x)
        for block in self.channel_blocks:
            x = block(x)
        for compressor in self.token_compressors:
            x = compressor(x)
        x = x.squeeze(-1)
        return self.classifier(x)

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    # RAGG accepts one intent per forward pass; keep sampling limits literal.
    batch_size = 1 if arg.intent is None or arg.max_samples > 0 else arg.batch_size
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()
    # 初始化分类器并加载权重
    classifier = ActionClassifier(input_channels=3)
    classifier.load_state_dict(torch.load("/home/zhenhua/grasp/OakInk-Grasp-Generation-main/pre_classification//best_classifier.pth"))  # 替换为你自己的路径
    classifier = DP(classifier).to(rank)
    classifier.eval()

    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")

    memory_dir = Path("memory")
    memory_bps = torch.from_numpy(np.load(memory_dir / "memory_bps.npy")).to(rank)
    memory_hand = torch.from_numpy(np.load(memory_dir / "memory_hand.npy")).to(rank)
    memory_intent = torch.from_numpy(np.load(memory_dir / "memory_id.npy")).to(rank)

    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 初始化每类的统计字典
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Testing")
        for step_idx, inp in enumerate(data_iter):
            if 0 < arg.max_samples <= step_idx:
                break
            inp = transform(inp)
            inp["memory_bps"] = memory_bps
            inp["memory_hand"] = memory_hand
            inp["memory_intent"] = memory_intent

            intent_name = arg.intent or inp[Queries.INTENT_NAME][0]
            prd, _ = model(inp=inp, step_idx=step_idx, mode="test", intent_name=intent_name)

            h_verts_r = prd["Refine.hand_verts"].detach()  # [B, 778, 3]
            # h_verts_r = inp[Queries.HAND_VERTS_OBJ].detach()  # [B, 778, 3]
            o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach()  # [B, 4096, 3]

            h_verts_gt = inp[Queries.HAND_VERTS_OBJ].detach()  # [B, 778, 3]

            features = torch.cat([h_verts_r, o_verts_ds], dim=1)  # [B, 4874, 3]
            logits = classifier(features)
            _, preds = torch.max(logits, dim=1)

            labels = torch.full_like(preds, intent_to_label[intent_name])

            # 统计每一类的正确数
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()

                class_stats[label]['total'] += 1
                if pred == label:
                    class_stats[label]['correct'] += 1

        # 打印结果
        total_correct = sum(v['correct'] for v in class_stats.values())
        total_samples = sum(v['total'] for v in class_stats.values())

        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Correct predictions: {total_correct}")
        logger.info(f"Overall Accuracy: {total_correct / total_samples:.4f}")

        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            acc = stats['correct'] / stats['total']
            logger.info(f"Class [{label_to_intent[label]}] - Correct: {stats['correct']}, Total: {stats['total']}, Acc: {acc:.4f}")


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold"],
                        help="specify when use IntGen model")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
