import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/robin/fareal/OakInk-Grasp-Generation-main/')

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
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
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
    classifier.load_state_dict(torch.load("/home/robin/fareal/OakInk-Grasp-Generation-main/classification//best_classifier.pth"))  # 替换为你自己的路径
    classifier = DP(classifier).to(rank)
    classifier.eval()

    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")

    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    # 初始化每类的统计字典
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    memory_bps = np.load('./memory_kmeans//memory_bps.npy')
    memory_hand = np.load('./memory_kmeans//memory_hand.npy')
    memory_id = np.load('./memory_kmeans//memory_id.npy')
    # memory_ov = np.load('./memory/memory_ov.npy')
    # memory_transl = np.load('./memory/memory_transl.npy')

    # 将NumPy数组转换为PyTorch Tensor
    memory_bps = torch.from_numpy(memory_bps)
    memory_hand = torch.from_numpy(memory_hand)
    memory_id = torch.from_numpy(memory_id)

    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Testing")
        for step_idx, inp in enumerate(data_iter):
            inp = transform(inp)
            inp['memory_bps'] = memory_bps
            inp['memory_hand'] = memory_hand
            inp['memory_intent'] = memory_id

            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test")  # GrabNet/HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test", intent_name=inp['intent_name'][0])  # IntGen

            h_verts_r = prd["Coarse.hand_verts"].detach()  # [B, 778, 3]
            o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach()  # [B, 4096, 3]

            features = torch.cat([h_verts_r, o_verts_ds], dim=1)  # [B, 4874, 3]
            logits = classifier(features)
            _, preds = torch.max(logits, dim=1)

            intent_name = inp['intent_name'][0]  # 因为是batch size=1，直接取第一个
            label = intent_to_label[intent_name]
            pred = preds[0].item()
            
            # 统计每一类的正确数
            class_stats[label]['total'] += 1
            if pred == label:
                class_stats[label]['correct'] += 1
            else:
                # 输出错误分类的信息
                logger.info(f"Instance {step_idx}: Wrong classification - Predicted: {label_to_intent[pred]}, Actual: {intent_name}")
                # 如果你想保存错误实例的索引，可以添加到一个列表中
                # wrong_indices.append(step_idx)

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
