import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

import torch.nn.functional as F

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
import open3d as o3d
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from lib.utils.transform import *


from oikit.oi_shape.utils import ALL_INTENT

from lib.datasets import create_dataset
from torch.utils.data.distributed import DistributedSampler

from pathlib import Path
import pandas as pd



# 颜色定义
COLOR_HAND = [0.82, 0.71, 0.55]  # 米黄色
COLOR_OBJECT = [1.0, 1.0, 0.0]   # 黄色

def is_valid_mesh(vertices):
    """检查顶点是否合法：非空且不含 NaN"""
    if len(vertices) == 0:
        return False
    if np.isnan(vertices).any():
        return False
    return True

output_dir = "/home/robin/fareal/OakInk-Grasp-Generation-main/joint_variant/"
output_path = os.path.join(output_dir, "sns_use_pen.xlsx")


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    split_to_cfg = {"train": cfg.DATASET.TRAIN, "val": cfg.DATASET.VAL, "test": cfg.DATASET.TEST}

    dataset = builder.build_dataset(split_to_cfg[arg.data_split], data_preset=cfg.DATA_PRESET)
    data_loader = DataLoader(dataset,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)


    train_data = builder.build_dataset(split_to_cfg['train'], data_preset=cfg.DATA_PRESET)






    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")


    ref_mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=cfg.DATA_PRESET.CENTER_IDX,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )



    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    logger.info("You are viz GrabNet's prediciton")


    obj_mesh = o3d.geometry.TriangleMesh()
    viz_ctx.add_geometry(obj_mesh)


    obj_mesh_right = o3d.geometry.TriangleMesh()  # 新增右边物体网格
    viz_ctx.add_geometry(obj_mesh_right)  # 添加到可视化上下文中

    memory_bps = np.load('./memory_kmeans//memory_bps.npy')
    memory_hand = np.load('./memory_kmeans//memory_hand.npy')
    memory_id = np.load('./memory_kmeans//memory_id.npy')
    # memory_ov = np.load('./memory/memory_ov.npy')
    # memory_transl = np.load('./memory/memory_transl.npy')

    # 将NumPy数组转换为PyTorch Tensor
    memory_bps = torch.from_numpy(memory_bps)
    memory_hand = torch.from_numpy(memory_hand)
    memory_id = torch.from_numpy(memory_id)

    all_angles = []

    model.eval()
    from lib.utils.pcd import point2point_signed
    with torch.no_grad():
        data_iter = etqdm(data_loader, desc="Processing samples", total=len(data_loader))
        for i, inp in enumerate(data_iter):
            # 前向推理
            inp = transform(inp)
            inp['memory_bps'] = memory_bps
            inp['memory_hand'] = memory_hand
            inp['memory_intent'] = memory_id
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")  # GrabNet, HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)  # IntGen


            compact_aa = rotmat_to_aa(prd["Coarse.fullpose"].detach())  # 假设输入形状为 [B, 16, 3, 3] 或 [B, 16*3]
            angles = torch.norm(compact_aa, dim=-1)      # [B, 16]
            all_angles.append(angles.cpu().numpy())

        
    all_angles = np.concatenate(all_angles, axis=0)  # [N_samples, 16]
    
    # 计算整体方差
    total_variance = np.var(all_angles)
    
    # 计算每个关节的统计量
    joint_stats = []
    for j in range(16):
        joint_angles = all_angles[:, j]
        joint_stats.append({
            "joint_index": j,
            "mean": np.mean(joint_angles),
            "variance": np.var(joint_angles),
            "min": np.min(joint_angles),
            "max": np.max(joint_angles),
            "std": np.std(joint_angles)
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(joint_stats)
    df.to_excel(output_path, index=False)




if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold","liftup"],
                        help="specify when use IntGen model")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)
