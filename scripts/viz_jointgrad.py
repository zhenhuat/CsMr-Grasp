import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib.cm import viridis

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


def compute_hand_saliency(classifier, hand_verts, obj_verts, target_class):
    """
    计算手部顶点对分类结果的梯度敏感度
    参数:
        classifier: 分类模型
        hand_verts: [B, 778, 3] 手部顶点
        obj_verts: [B, N, 3] 物体顶点
        target_class: [B,] 目标类别
    返回:
        hand_grads: [B, 778] 每个手部顶点的梯度重要性
    """
    # 只对手部顶点计算梯度
    hand_verts.requires_grad = True
    
    # 固定物体顶点不计算梯度
    with torch.enable_grad():
        obj_verts_fixed = obj_verts.detach()
    
    # 拼接特征
        features = torch.cat([hand_verts, obj_verts_fixed], dim=1)
    
    # 计算分类logits
    logits = classifier(features)
    
    # 创建one-hot向量
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, target_class.view(-1, 1), 1)
    
    # 清除之前的梯度
    classifier.zero_grad()
    if hand_verts.grad is not None:
        hand_verts.grad.zero_()
    
    # 反向传播计算梯度
    logits.backward(gradient=one_hot, retain_graph=True)
    
    # 获取手部顶点梯度绝对值
    hand_grads = hand_verts.grad.data.abs().sum(dim=-1)  # [B, 778]
    
    return hand_grads


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=True,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")

    # 初始化分类器
    classifier = ActionClassifier().to("cuda")
    try:
        classifier.load_state_dict(torch.load("/home/robin/fareal/OakInk-Grasp-Generation-main/classification/best_classifier.pth"))
        classifier.eval()
        logger.info("Classifier loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load classifier: {str(e)}")

    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    logger.info("Visualizing hand vertices saliency with ground truth intent")
    model.eval()

    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        for i, inp in enumerate(data_iter):
            inp = transform(inp)
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=inp['intent_name'][0])

            print('Current manipulation:', inp['intent_name'])

            h_verts_prd = prd["Refine.hand_verts"].to("cuda")  # [B, 778, 3]
            o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].to("cuda")  # [B, N, 3]
            intent_vec = inp[Queries.INTENT_VEC].to("cuda")  # [B, 2]
            cls_labels = torch.argmax(intent_vec, dim=1)  # [B,]

            try:
                # 计算手部顶点梯度
                hand_grads = compute_hand_saliency(classifier, h_verts_prd, o_verts_ds, cls_labels)
                
                # 归一化梯度值用于可视化
                hand_grads_norm = (hand_grads - hand_grads.min()) / (hand_grads.max() - hand_grads.min() + 1e-8)
                
                # 准备可视化数据
                hand_points = h_verts_prd[0].cpu().numpy()
                hand_colors = (viridis(hand_grads_norm[0].cpu().numpy())[:, :3] * 255).astype(np.uint8)
                hand_colors = hand_colors.astype(np.uint8)
                
                # 准备物体数据(灰色显示)
                obj_points = o_verts_ds[0].cpu().numpy()
                obj_colors = np.full((len(obj_points), 3), 150, dtype=np.uint8)
                
                # 更新可视化
                viz_ctx.update_by_pc("hand_saliency", hand_points, pcolors=hand_colors)
                viz_ctx.update_by_pc("object", obj_points, pcolors=obj_colors)
                
                # 打印信息
                logger.info(f"Intent: {inp['intent_name'][0]} | "
                           f"Max saliency: {hand_grads.max().item():.4f} | "
                           f"Mean saliency: {hand_grads.mean().item():.4f}")
                
            except Exception as e:
                logger.warning(f"Saliency computation error: {str(e)}")
            
            # 等待用户按键
            show_next = False
            while not show_next:
                viz_ctx.step()


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUs")

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