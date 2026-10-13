import argparse
import os
from time import time
from argparse import Namespace

import numpy as np
import torch
import matplotlib.pyplot as plt
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
import open3d as o3d

from matplotlib.colors import LinearSegmentedColormap

from scipy.spatial.transform import Rotation as R
import open3d as o3d
import pandas as pd
# from lib.utils.transform import *

# 自定义颜色映射（暗红 → 浅绿）
colors_255 = np.array([
    [198,  91,  63],   # 暗红（High Gradient）
    [222, 160, 145],   # 橙红
    [237, 204, 197],   # 叶黄
    [119, 165, 179],   # 淡黄
    [215, 230, 234],   # 浅绿（Low Gradient）
])
colors = colors_255 / 255.0
custom_cmap = LinearSegmentedColormap.from_list("custom_red_to_blue", colors)



# 扩展 VizContext 类以支持 flags 状态管理
from lib.viztools.viz_o3d_utils import VizContext
class MyVizContext(VizContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = {}  # 添加状态字典

    def toggle_flag(self, key):
        self.flags[key] = not self.flags.get(key, False)

def add_noise_to_hand_pose(
    hand_pose: torch.Tensor,
    normalized_grad: np.ndarray,
    noise_scale: float = 1.0,
    device: str = "cuda"
):
    """
    给 hand_pose 添加基于梯度大小的噪声扰动（axis-angle 表示）
    
    参数:
        hand_pose (torch.Tensor): [B, 15, 3] 原始手部姿态参数（轴角表示）
        normalized_grad (np.ndarray): [16, ] 每个关节点的归一化梯度值（wrist + 15 joints）
        noise_scale (float): 控制最大噪声强度（弧度）
        device (str): 计算设备（"cuda" 或 "cpu"）

    返回:
        noised_hand_pose (torch.Tensor): [B, 15, 3] 带噪声的 pose 参数
    """
    hand_pose = hand_pose.reshape(-1,16,3)
    B, N_JOINTS, _ = hand_pose.shape

    # print(normalized_grad.shape)
    # print(hand_pose.shape)
    # exit()
    normalized_grad = torch.from_numpy(normalized_grad).float().to(device=device)  # [15]
    normalized_grad = normalized_grad.unsqueeze(0).unsqueeze(-1)  # [1, 15, 1]

    # 构造噪声（标准正态 × 权重 × scale）
    # noise = torch.randn_like(hand_pose) * noise_scale * normalized_grad
    noise = torch.randn_like(hand_pose) * 0.3
    # 添加噪声
    noised_hand_pose = hand_pose + noise

    return noised_hand_pose.reshape(B,-1)

def save_hand_poses_to_csv(model, data_loader, rank, transform, output_file="hand_poses.csv"):
    """
    遍历数据集，提取每个样本的 hand_pose_obj（16x3）和 intent_name，
    并以关节名称命名列，保存为 CSV 文件。
    
    参数:
        model: GrabNet/IntGen 模型（已设置为 eval 模式）
        data_loader: 数据加载器
        rank: 设备编号（GPU ID）
        transform: 输入变换函数
        output_file: 输出文件路径
    """

        # 定义 MANO 关节名称（用于 CSV 列名）
    MANO_JOINT_NAMES_16 = [
        'wrist',
        'thumb_CMC', 'thumb_MCP', 'thumb_IP',
        'index_MCP', 'index_PIP', 'index_DIP',
        'middle_MCP', 'middle_PIP', 'middle_DIP',
        'ring_MCP', 'ring_PIP', 'ring_DIP',
        'pinky_MCP', 'pinky_PIP', 'pinky_DIP'
    ]

    all_samples = []

    model.eval()
    with torch.no_grad():
        data_iter = etqdm(data_loader, desc=f"Processing samples", leave=False)
        for i, inp in enumerate(data_iter):
            # 数据变换
            inp = transform(inp)

            # 提取 intent_name
            intent_name = inp['intent_name'][0]

            # 提取 hand_pose_obj: [1, 48] -> [16, 3]
            hand_pose = inp['hand_pose_obj'].squeeze(0).cpu().numpy()  # shape: [48]
            hand_pose_reshaped = hand_pose.reshape(16, 3)  # reshape to [16, 3]

            # 构建 DataFrame 行，使用 MANO 关节名称
            sample_data = {
                "sample_id": i,
                "intent_name": intent_name,
                **{f"{MANO_JOINT_NAMES_16[jid]}_axis_{dim}": hand_pose_reshaped[jid, dim]
                   for jid in range(16) for dim in range(3)}
            }
            all_samples.append(sample_data)

    # 保存为 CSV
    df = pd.DataFrame(all_samples)
    df.to_csv(output_file, index=False)
    print(f"✅ Hand poses saved to {output_file}")



def save_all_gradients_to_csv(model, data_loader, rank,transform, output_file="gradients.csv"):
    """
    遍历数据集，提取每个样本的 prd_grad，并计算每个关节的模长，保存为 CSV 文件
    
    参数:
        model: GrabNet/IntGen 模型（已设置为 eval 模式）
        data_loader: 数据加载器
        rank: 设备编号（GPU ID）
        output_file: 输出文件路径
    """

    # 定义 MANO 关节名称（用于 CSV 列名）
    MANO_JOINT_NAMES_16 = [
        'wrist',
        'thumb_CMC', 'thumb_MCP', 'thumb_IP',
        'index_MCP', 'index_PIP', 'index_DIP',
        'middle_MCP', 'middle_PIP', 'middle_DIP',
        'ring_MCP', 'ring_PIP', 'ring_DIP',
        'pinky_MCP', 'pinky_PIP', 'pinky_DIP'
    ]

    all_samples = []

    model.eval()
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        for i, inp in enumerate(data_iter):
            # 推理获取 prd_grad
            inp = transform(inp)
            prd, _ = model(inp=inp, step_idx=i, mode="test")
            grad_tensor = prd['prd_grad'].detach().cpu().squeeze(0)  # shape: [16,6]

            # 计算每个关节点的梯度模长
            joint_grad_magnitude = torch.norm(grad_tensor, dim=1).cpu().numpy()  # [16, ]
            # if i == 4:
            #     exit()
            # 构建 DataFrame 行
            sample_data = {
                "sample_id": i,
                **{name: joint_grad_magnitude[jid] for jid, name in enumerate(MANO_JOINT_NAMES_16)}
            }
            all_samples.append(sample_data)

    # 保存为 CSV
    df = pd.DataFrame(all_samples)
    df.to_csv(output_file, index=False)
    print(f"✅ Gradients saved to {output_file}")

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


    viz_ctx = MyVizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    obj_mesh = o3d.geometry.TriangleMesh()
    viz_ctx.add_geometry(obj_mesh)
    obj_mesh_right = o3d.geometry.TriangleMesh()
    viz_ctx.add_geometry(obj_mesh_right)

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    viz_ctx.register_key_callback("G", lambda _: viz_ctx.toggle_flag("use_gradient_color"))
    logger.info("You are visualizing GrabNet's prediction")
    model.eval()

    from manotorch.utils.anchorutils import anchor_load_driver, get_region_palm_mask

    results = anchor_load_driver("./assets")  # 替换为你自己的路径
    palm = results[0]
    hand_region_assignment = results[1]
    merged_vertex_assignment = results[2]

    colormap = custom_cmap

    # all_gradients = []


    # gradient_csv_path = "all_sample_gradients.csv"
    # save_all_gradients_to_csv(model, data_loader, rank,transform, gradient_csv_path)
    # object_csv_path = "data_rotation/all.csv"
    # save_hand_poses_to_csv(model, data_loader, rank,transform, object_csv_path)
    # exit()

    from lib.utils.pcd import point2point_signed
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")

        for i, inp in enumerate(data_iter):
            inp = transform(inp)
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")  # GrabNet/HoverGen
            else:
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)  # IntGen inp['intent_name'][0]

            # print(inp[Queries.HAND_POSE_OBJ])

            print('current manipulation is:', inp['intent_name'])

            # 获取当前样本的梯度数据 [16,6]
            grad_tensor = prd['prd_grad'].clone().detach().cpu().squeeze(0)  # shape: [16,6]
            # print(grad_tensor)

            # 计算每个关节点的梯度模长 [16]
            joint_grad_magnitude = torch.norm(grad_tensor, dim=1).numpy()
            # print(joint_grad_magnitude)

            # print(grad_tensor)
            # exit()

            # 使用 log 处理防止手腕主导
            joint_grad_magnitude += 1e-8
            log_grad = np.log(joint_grad_magnitude)


            # 分位数归一化
            quantiles = np.percentile(log_grad, [25, 50, 75])
            normalized_grad = np.interp(log_grad, quantiles, [0, 0.5, 1])


            # 反向归一化：梯度越小 → 噪声越大
            inverted_normalized = 1.0 - normalized_grad  # shape: [16, ]

            # 映射为颜色 [16,3]
            joint_colors = colormap(normalized_grad)[:, :3]  # [16,3]

            # 初始化顶点颜色为灰色 [778,3]
            vertex_colors = np.tile([0.82, 0.71, 0.55], (778, 1))

            # 使用 GrabNet 的锚点配置给每个区域上色
            for joint_idx in range(16):  # 遍历所有 16 个关节点
                mask = get_region_palm_mask(joint_idx, None, merged_vertex_assignment, None)
                vertex_colors[mask] = joint_colors[joint_idx]


            # 获取 Refine 阶段的 pose 参数
            global_orient_new = prd["Refine.global_orient"].clone().detach().to(rank)  # [B, 3]
            hand_pose_new = prd["Refine.hand_pose"].clone().detach().to(rank)          # [B, 45]
            

            mano_shape_new = inp[Queries.HAND_SHAPE].clone().detach().cpu()         # [B, 10]
            with torch.no_grad():
                mano_pose_new = torch.cat([global_orient_new, hand_pose_new], dim=1)  # (B, 48)

                # mano_pose_new = add_noise_to_hand_pose(mano_pose_new,inverted_normalized).clone()
                # print(mano_pose.device,mano_shape.device)
                # exit()
                mano_pose_new = mano_pose_new.detach().cpu()
                mano_pose_new = mano_layer(mano_pose_new, mano_shape_new)
                verts_rhand_noise = mano_pose_new.verts + prd[f"Refine.transl"].clone().unsqueeze(1).detach().cpu()



            # 准备可视化数据
            h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()  # [B,778,3]
            h_faces = mano_layer.th_faces.detach().cpu().numpy()  # [1538,3]
            h_verts_gt = inp[Queries.HAND_VERTS_OBJ].detach().cpu().numpy()
            o_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS].detach().cpu().numpy()
            o_norms_ds = inp[Queries.OBJ_NORMALS_OBJ_DS].detach().cpu().numpy()
            my_o_verts = inp[Queries.OBJ_VERTS_OBJ].detach().cpu().numpy()

            if hasattr(inp, Queries.ALT_HAND_VERTS_OBJ):
                alt_hand_verts = inp[Queries.ALT_HAND_VERTS_OBJ].detach().cpu().numpy()

            n_items = h_verts_r.shape[0]
            viz_ofs = np.array([0.2, 0.0, 0.0])

            # 转为 numpy 用于 viz
            verts_rhand_noise = verts_rhand_noise.cpu().numpy()

            for i in range(n_items):
                show_next = False

                # 更新 mesh（原始 + 加噪）
                viz_ctx.update_by_mesh("hand_clean", verts_rhand_noise[i], h_faces, vcolors=[0.5, 0.5, 0.5])
                # viz_ctx.update_by_mesh("hand_noised", h_verts_gt[i]+ 2 * viz_ofs, h_faces, vcolors=[0.82, 0.71, 0.55])

                # 物体部分保持不变
                ov = o_verts_ds[i]
                on = o_norms_ds[i]
                # viz_ctx.update_by_pc("obj", ov, on, pcolors=CMap["lime"])
                # viz_ctx.update_by_pc("obj_ofs", ov + 2 * viz_ofs, on, pcolors=CMap["lime"])

                # 渲染物体 mesh
                my_ov = my_o_verts[i]
                o_faces = inp[Queries.OBJ_FACES].detach().cpu().numpy().squeeze(0)
                obj_mesh.vertices.clear()
                obj_mesh.triangles.clear()
                
                # obj_mesh.vertices = o3d.utility.Vector3dVector(my_ov)
                # obj_mesh.triangles = o3d.utility.Vector3iVector(o_faces)
                # obj_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.0]] * len(my_ov)))
                # obj_mesh.compute_vertex_normals()
                # obj_mesh.compute_triangle_normals()
                # viz_ctx.update_geometry(obj_mesh)

                obj_mesh_right.vertices.clear()
                obj_mesh_right.triangles.clear()
                obj_mesh_right.vertices = o3d.utility.Vector3dVector(my_ov + 2 * viz_ofs)
                obj_mesh_right.triangles = o3d.utility.Vector3iVector(o_faces)
                obj_mesh_right.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.0]] * len(my_ov)))
                obj_mesh_right.compute_vertex_normals()
                obj_mesh_right.compute_triangle_normals()
                viz_ctx.update_geometry(obj_mesh_right)

                while not show_next:
                    viz_ctx.step()


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