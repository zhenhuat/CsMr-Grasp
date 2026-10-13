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

    all_data = {
        'attentions': [],  # 存储注意力矩阵
        'intents': [],     # 存储意图标签
        'sample_ids': []   # 存储样本ID (如果有的话)
    }



    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Testing")
        for step_idx, inp in enumerate(data_iter):
            inp = transform(inp)

            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test")  # GrabNet/HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test", intent_name=inp['intent_name'][0])  # IntGen

            # 存储数据
            attention_list = prd["Refine.attention"]
            attention_numpy = [att.detach().cpu().numpy() if torch.is_tensor(att) else np.array(att) 
                          for att in attention_list]
            all_data['attentions'].append(attention_numpy)
            all_data['intents'].append(inp['intent_name'][0])
            
            # 如果有样本ID的话
            if 'sample_id' in inp:
                all_data['sample_ids'].append(inp['sample_id'][0])
            else:
                all_data['sample_ids'].append(step_idx)

    # 转换为numpy数组
    save_dict = {
        'attentions': np.array(all_data['attentions']),
        'intents': np.array(all_data['intents']),
        'sample_ids': np.array(all_data['sample_ids'])
    }

    # 保存为npz文件
    np.savez("attention_data.npz", **save_dict)
    logger.info(f"Saved attention data to attention_data.npz")
    logger.info(f"Attention matrices shape: {save_dict['attentions'].shape}")


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
