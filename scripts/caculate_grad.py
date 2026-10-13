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

import  matplotlib.pyplot as plt

import torch.nn as nn
from collections import defaultdict
# -----------------------------


MANO_JOINT_NAMES_16 = [
    'wrist',                     # 0: 手腕（全局旋转）
    'thumb CMC', 'thumb MCP', 'thumb IP',  # 1-3: 拇指（CMC, MCP, IP）
    'index MCP', 'index PIP', 'index DIP',  # 4-6: 食指（MCP, PIP, DIP）
    'middle MCP', 'middle PIP', 'middle DIP',  # 7-9: 中指（MCP, PIP, DIP）
    'ring MCP', 'ring PIP', 'ring DIP',  # 10-12: 无名指（MCP, PIP, DIP）
    'pinky MCP', 'pinky PIP', 'pinky DIP'  # 13-15: 小指（MCP, PIP, DIP）
]

MANO_JOINT_NAMES_16_CN = [
    '手腕',                      # 0: 手腕（全局旋转）
    '拇指根部（CMC）', '拇指掌指（MCP）', '拇指指间（IP）',  # 1-3: 拇指
    '食指掌指（MCP）', '食指近端指间（PIP）', '食指远端指间（DIP）',  # 4-6: 食指
    '中指掌指（MCP）', '中指近端指间（PIP）', '中指远端指间（DIP）',  # 7-9: 中指
    '无名指掌指（MCP）', '无名指近端指间（PIP）', '无名指远端指间（DIP）',  # 10-12: 无名指
    '小指掌指（MCP）', '小指近端指间（PIP）', '小指远端指间（DIP）'  # 13-15: 小指
]

def save_gradient_stats(grad_tensor, output_dir="outputs"):
    """
    grad_tensor: shape [B, 16, 6]
    统计所有样本的梯度模长，计算平均值，并用柱状图展示（横轴为关节名称）
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "viz"), exist_ok=True)

    num_samples = grad_tensor.shape[0]
    all_grad_norms = []

    # 遍历每个样本，计算梯度模长
    for sample_idx in range(num_samples):
        grad = grad_tensor[sample_idx].cpu().numpy()  # [16, 6]
        grad_norm = np.linalg.norm(grad, axis=1)      # [16, ]
        all_grad_norms.append(grad_norm)

    all_grad_norms = np.array(all_grad_norms)  # shape: [N, 16]

    # 保存原始梯度数据
    np.save(os.path.join(output_dir, "gradients.npy"), all_grad_norms)

    # 计算平均梯度
    avg_gradient = np.mean(all_grad_norms, axis=0)
    np.save(os.path.join(output_dir, "avg_gradient.npy"), avg_gradient)

    # 标准化（Z-score）
    mean = np.mean(all_grad_norms, axis=0)
    std = np.std(all_grad_norms, axis=0)
    norm_gradients = (all_grad_norms - mean) / (std + 1e-8)
    np.save(os.path.join(output_dir, "norm_gradients.npy"), norm_gradients)

    # 绘制柱状图（使用 MANO 关节名称作为横坐标）
    joint_indices = np.arange(16)
    plt.figure(figsize=(14, 6))
    bars = plt.bar(joint_indices, avg_gradient, color='skyblue', edgecolor='black')
    plt.xticks(joint_indices, MANO_JOINT_NAMES_16, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Average Gradient Magnitude per Joint")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(os.path.join(output_dir, "viz", "average_gradient_bar.png"))
    plt.close()

    print(f"✅ Gradient stats saved to {output_dir}")




def save_gradient_stats_samples(grad_tensor, output_dir="outputs"):
    """
    grad_tensor: shape [B, 16, 6]
    统计所有样本的梯度模长，计算平均值，并用柱状图展示（横轴为关节名称）
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "viz"), exist_ok=True)

    num_samples = grad_tensor.shape[0]
    all_grad_norms = []

    # 遍历每个样本，计算梯度模长
    for sample_idx in range(num_samples):
        grad = grad_tensor[sample_idx].cpu().numpy()  # [16, 6]
        grad_norm = np.linalg.norm(grad, axis=1)      # [16, ]
        all_grad_norms.append(grad_norm)

    all_grad_norms = np.array(all_grad_norms)  # shape: [N, 16]

    # 保存原始梯度数据
    np.save(os.path.join(output_dir, "gradients.npy"), all_grad_norms)

    # 计算平均梯度
    avg_gradient = np.mean(all_grad_norms, axis=0)
    np.save(os.path.join(output_dir, "avg_gradient.npy"), avg_gradient)

    # 标准化（Z-score）
    mean = np.mean(all_grad_norms, axis=0)
    std = np.std(all_grad_norms, axis=0)
    norm_gradients = (all_grad_norms - mean) / (std + 1e-8)
    np.save(os.path.join(output_dir, "norm_gradients.npy"), norm_gradients)

    # 绘制柱状图（使用 MANO 关节名称作为横坐标）
    joint_indices = np.arange(16)
    plt.figure(figsize=(14, 6))
    bars = plt.bar(joint_indices, avg_gradient, color='skyblue', edgecolor='black')
    plt.xticks(joint_indices, MANO_JOINT_NAMES_16, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Average Gradient Magnitude per Joint")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(os.path.join(output_dir, "viz", "average_gradient_bar.png"))
    plt.close()

    print(f"✅ Gradient stats saved to {output_dir}")


def save_gradient_csv(grad_tensor, output_dir="outputs"):
    """
    grad_tensor: shape [B, 16, 6]
    只保存梯度数据到本地，不做任何绘图
    """
    os.makedirs(output_dir, exist_ok=True)

    num_samples = grad_tensor.shape[0]
    all_grad_norms = []

    # 遍历每个样本，计算梯度模长
    for sample_idx in range(num_samples):
        grad = grad_tensor[sample_idx].cpu().numpy()  # [16, 6]
        grad_norm = np.linalg.norm(grad, axis=1)       # [16, ]
        print(grad_norm)
        exit()
        all_grad_norms.append(grad_norm)

    all_grad_norms = np.array(all_grad_norms)  # shape: [N, 16]

    # 保存原始梯度数据
    np.save(os.path.join(output_dir, "gradients.npy"), all_grad_norms)

    # 计算平均梯度
    avg_gradient = np.mean(all_grad_norms, axis=0)
    np.save(os.path.join(output_dir, "avg_gradient.npy"), avg_gradient)

    # 标准化（Z-score）
    mean = np.mean(all_grad_norms, axis=0)
    std = np.std(all_grad_norms, axis=0)
    norm_gradients = (all_grad_norms - mean) / (std + 1e-8)
    np.save(os.path.join(output_dir, "norm_gradients.npy"), norm_gradients)

    # 可选：保存为 CSV 文件
    import pandas as pd
    df = pd.DataFrame(all_grad_norms, columns=MANO_JOINT_NAMES_16)
    df.to_csv(os.path.join(output_dir, "gradients.csv"), index=False)

    df_avg = pd.DataFrame([avg_gradient], columns=MANO_JOINT_NAMES_16)
    df_avg.to_csv(os.path.join(output_dir, "avg_gradient.csv"), index=False)

    df_norm = pd.DataFrame(norm_gradients, columns=MANO_JOINT_NAMES_16)
    df_norm.to_csv(os.path.join(output_dir, "norm_gradients.csv"), index=False)

    print(f"✅ Gradient data saved to {output_dir}")

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

    classifier = ActionClassifier(input_channels=3)
    classifier.load_state_dict(torch.load("/home/robin/fareal/OakInk-Grasp-Generation-main/classification//best_classifier.pth"))
    classifier = DP(classifier).to(rank)
    classifier.eval()

    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")

    intent_to_label = {'use': 0, 'hold': 1}
    label_to_intent = {v: k for k, v in intent_to_label.items()}

    # 存储所有样本的梯度
    all_gradients = []

    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Testing")
        for step_idx, inp in enumerate(data_iter):
            inp = transform(inp)

            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test")  # GrabNet/HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=step_idx, mode="test", intent_name=inp['intent_name'][0])  # IntGen

            # 获取关节点旋转梯度: [B, 16, 6]
            grad_tensor = prd['prd_grad'].detach()  # 替换为你实际的梯度变量名
            joint_grad_magnitude = torch.norm(grad_tensor.cpu().squeeze(0), dim=1).numpy()
            # print(grad_tensor)
            exit()
            all_gradients.append(grad_tensor)

        # 将所有样本拼接
        all_gradients = torch.cat(all_gradients, dim=0)

        # 调用函数进行统计、保存、可视化
        # save_gradient_stats(all_gradients, output_dir="outputs")
        save_gradient_csv(all_gradients, output_dir="outputs")



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