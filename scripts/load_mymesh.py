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
sys.path.append('/home/robin/fareal/icassp_grasp/')

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


from oikit.oi_shape.utils import ALL_INTENT

from lib.datasets import create_dataset
from torch.utils.data.distributed import DistributedSampler

from pathlib import Path

# 创建保存路径
save_dir = Path("./mymesh/CsMr_unseen/use/")
save_dir.mkdir(parents=True,exist_ok=True)


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



            # h_verts_r = prd["Coarse.hand_verts"].detach()  # [B, 778, 3]
            h_verts_r = prd["Refine.hand_verts"].detach()  # [B, 778, 3]
            # h_verts_r = inp[Queries.HAND_VERTS_OBJ].detach()  # [B, 778, 3]
            # o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach()  # [B, 4096, 3]
            o_verts_ds = inp['obj_verts_obj_ds'].detach().to(h_verts_r.device)


            # 获取数据
            h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy().astype(np.float64)
            my_o_verts = inp[Queries.OBJ_VERTS_OBJ].detach().cpu().numpy().astype(np.float64)
            # my_o_verts = inp['verts_object'].detach().cpu().numpy()

            o_faces = inp[Queries.OBJ_FACES].detach().cpu().numpy().squeeze(0)
            h_faces = mano_layer.th_faces.detach().cpu().numpy()

            n_items = h_verts_r.shape[0]

            for j in range(n_items):
                hand_v = h_verts_r[j]
                obj_v = my_o_verts[j]

                # 检查顶点是否有效
                if not is_valid_mesh(hand_v) or not is_valid_mesh(obj_v):
                    print(f"Skipping invalid mesh at batch {i}, index {j}")
                    continue

                # 构建手部网格
                hand_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(hand_v),
                    triangles=o3d.utility.Vector3iVector(h_faces)
                )
                hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.array([COLOR_HAND] * len(hand_v))
                )

                # 构建设备网格
                obj_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(obj_v),
                    triangles=o3d.utility.Vector3iVector(o_faces)
                )
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.array([COLOR_OBJECT] * len(obj_v))
                )

                # 合并网格
                combined_mesh = hand_mesh + obj_mesh

                # 保存文件
                save_path = save_dir / f"sample_{i * n_items + j:04d}.obj"
                o3d.io.write_triangle_mesh(str(save_path), combined_mesh)

                logger.info(f"Saved combined mesh to {save_path}")



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
