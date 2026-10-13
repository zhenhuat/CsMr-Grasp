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

colors_255 = np.array([
    [198,  91,  63],   # 暗红（Dark Red）
    [222,  160,   145],   # 橙红（OrangeRed）
    [237, 204,  197],   # 叶黄/黄绿（YellowGreen）
    [119, 165, 179],   # 淡黄（Light Yellow）
    [215, 230, 234],   # 浅绿（Light Green）
])

colors = colors_255 / 255.0

custom_cmap = LinearSegmentedColormap.from_list("pink_to_blue", colors)

# 扩展 VizContext 类以支持 flags 状态管理
class MyVizContext(VizContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = {}  # 添加状态字典

    def toggle_flag(self, key):
        self.flags[key] = not self.flags.get(key, False)


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


    obj_mesh_right = o3d.geometry.TriangleMesh()  # 新增右边物体网格
    viz_ctx.add_geometry(obj_mesh_right)  # 添加到可视化上下文中

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    viz_ctx.register_key_callback("G", lambda _: viz_ctx.toggle_flag("use_gradient_color"))
    logger.info("You are visualizing GrabNet's prediction")
    model.eval()

    # 加载 GrabNet 锚点配置
    from manotorch.utils.anchorutils import anchor_load_driver, get_region_palm_mask

    results = anchor_load_driver("./assets")  # 替换为你自己的路径
    palm = results[0]
    hand_region_assignment = results[1]
    merged_vertex_assignment = results[2]

    # 创建颜色映射器
    # colormap = plt.get_cmap('viridis')
    colormap = custom_cmap

    from lib.utils.pcd import point2point_signed
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0}")
        for i, inp in enumerate(data_iter):
            inp = transform(inp)
            if arg.intent is None:
                prd, _ = model(inp=inp, step_idx=i, mode="test")  # GrabNet/HoverGen
            else:
                # prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=inp['intent_name'][0])  # IntGen
                prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)  # IntGen

            print('current manipulation is:', inp['intent_name'])

            # 获取当前样本的梯度数据 [16,6]
            grad_tensor = prd['prd_grad'].detach().cpu().squeeze(0)  # shape: [16,6]

            # print(grad_tensor)
            # exit()

            rotation_grad = grad_tensor[:, 1]  # 取每行的第一个元素
            # softmax_grad = torch.softmax(rotation_grad*10e13, dim=0).numpy()
            normalized_grad = (rotation_grad - rotation_grad.min()) / (rotation_grad.max() - rotation_grad.min())
            print(normalized_grad)
            # exit()
            # 映射到颜色cmap上
            colormap = plt.get_cmap('inferno')  # 可以选择其他colormap如'plasma', 'inferno'等
            joint_colors = colormap(normalized_grad)[:, :3]


            # 计算每个关节点的梯度模长 [16]
            # joint_grad_magnitude = torch.norm(grad_tensor, dim=1).numpy()

            # 使用 log 处理防止手腕主导
            # joint_grad_magnitude += 1e-8
            # log_grad = np.log(joint_grad_magnitude)

            # # 使用分位数归一化
            # quantiles = np.percentile(log_grad, [25, 50, 75])
            # normalized_grad = np.interp(log_grad, quantiles, [0, 0.5, 1])

            # # 映射为平滑渐变颜色（暖色调表示高梯度值）
            # # colormap = plt.get_cmap('cividis')  # 反转颜色映射，红色表示高梯度
            # joint_colors = colormap(normalized_grad)[:, :3]

            # 初始化顶点颜色为灰色 [778,3]
            # vertex_colors = np.tile([0.5, 0.5, 0.5], (778, 1))
            vertex_colors = np.tile([0.82, 0.71, 0.55], (778, 1))

            # 使用 GrabNet 的锚点配置给每个区域上色
            for joint_idx in range(16):  # 遍历所有 16 个关节点
                mask = get_region_palm_mask(joint_idx, None, merged_vertex_assignment, None)
                # print(joint_idx,mask)

                # mask = get_region_palm_mask(joint_idx, palm, merged_vertex_assignment, hand_region_assignment)
                # mask = (merged_vertex_assignment == joint_idx)
                vertex_colors[mask] = joint_colors[joint_idx]

            # print(vertex_colors)
            # exit()

            # print("Colored vertices count:", np.sum(vertex_colors != [0.5, 0.5, 0.5]))

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

            for i in range(n_items):
                show_next = False

                # 切换显示模式（默认不显示梯度颜色）
                # if viz_ctx.flags.get("use_gradient_color", False):
                #     current_colors = vertex_colors
                # else:
                #     current_colors = CMap["deepskyblue"]

                current_colors = vertex_colors

                # 可视化带颜色的手部 mesh
                
                # viz_ctx.update_by_mesh("hand_coarse", h_verts_r[i], h_faces, vcolors=[0.82, 0.71, 0.55])  #肉色
                viz_ctx.update_by_mesh("hand_coarse", h_verts_r[i], h_faces, vcolors=vertex_colors)

                # 其他部分保持不变
                # viz_ctx.update_by_mesh("hand_refine", h_verts_gt[i] + 2 * viz_ofs, h_faces, vcolors=[0.82, 0.71, 0.55])
                ov = o_verts_ds[i]
                on = o_norms_ds[i]
                # viz_ctx.update_by_pc("obj", ov, on, pcolors=CMap["lime"])
                # viz_ctx.update_by_pc("obj_ofs", ov + 2 * viz_ofs, on, pcolors=CMap["lime"])

                my_ov = my_o_verts[i]
                obj_mesh.vertices.clear()
                obj_mesh.triangles.clear()

                o_faces = inp[Queries.OBJ_FACES].detach().cpu().numpy().squeeze(0)
                obj_mesh.vertices = o3d.utility.Vector3dVector(my_ov)
                obj_mesh.triangles = o3d.utility.Vector3iVector(o_faces)
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.0]] * len(my_ov)))
                obj_mesh.compute_vertex_normals()  # 可选，如果你需要更精确的光照效果
                obj_mesh.compute_triangle_normals()
                viz_ctx.update_geometry(obj_mesh)

                ####viz_reference hand
                # viz_ctx.add_geometry(obj_mesh)
                # obj_mesh_right.vertices.clear()
                # obj_mesh_right.triangles.clear()
                # obj_mesh_right.vertices = o3d.utility.Vector3dVector(my_ov + 2 *viz_ofs)
                # obj_mesh_right.triangles = o3d.utility.Vector3iVector(o_faces)
                # obj_mesh_right.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.0]] * len(my_ov)))
                # obj_mesh_right.compute_vertex_normals()
                # obj_mesh_right.compute_triangle_normals()
                # viz_ctx.update_geometry(obj_mesh_right)


                if hasattr(inp, Queries.ALT_HAND_VERTS_OBJ):
                    viz_ctx.update_by_mesh("alt_hand", alt_hand_verts[i], h_faces, vcolors=CMap["blue"])
                    viz_ctx.update_by_mesh("alt_hand_ofs", alt_hand_verts[i] + viz_ofs, h_faces, vcolors=CMap["blue"])

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