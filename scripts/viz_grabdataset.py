import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
import sys
sys.path.append('/home/robin/fareal/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_data import grasp_data_collate
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.etqdm import etqdm
from torch.nn.parallel import DataParallel as DP

from manotorch.manolayer import ManoLayer
from lib.viztools.viz_o3d_utils import VizContext
import open3d as o3d

# 颜色映射
CMap = {
    "red": np.array([1.0, 0.0, 0.0]),
    "green": np.array([0.0, 1.0, 0.0]),
    "blue": np.array([0.0, 0.0, 1.0]),
    "yellow": np.array([1.0, 1.0, 0.0]),
    "cyan": np.array([0.0, 1.0, 1.0]),
    "magenta": np.array([1.0, 0.0, 1.0]),
    "lime": np.array([0.0, 1.0, 0.0]),
    "tomato": np.array([255, 99, 71]) / 255.0,
    "orange": np.array([255, 165, 0]) / 255.0,
    "white": np.array([255, 255, 255]) / 255.0,
    "black": np.array([0, 0, 0]) / 255.0,
}

class LoadData(Dataset):
    def __init__(self, dataset_dir, ds_name='train', dtype=torch.float32, only_params=False, load_on_ram=False):
        super().__init__()
        self.only_params = only_params
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path, 'grabnet_%s.npz' % ds_name))

        frame_names = np.load(os.path.join(dataset_dir, ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        bps_fname = os.path.join(dataset_dir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs = torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        self.load_on_ram = load_on_ram

        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

    def _np2torch(self, ds_path):
        data = np.load(ds_path, allow_pickle=True)
        return {k: torch.tensor(data[k]) for k in data.files}

    def load_disk(self, idx):
        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])
        return default_collate([self._np2torch(f) for f in self.frame_names[idx]])

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]

    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params and not self.load_on_ram:
            data_out.update(self.load_disk(idx))
        return data_out

class SimpleMesh:
    def __init__(self, filename=None, vc=None):
        self.vertices = None
        self.faces = None
        self.vertex_colors = None

        if filename:
            mesh = o3d.io.read_triangle_mesh(filename)
            self.vertices = np.asarray(mesh.vertices)
            self.faces = np.asarray(mesh.triangles)

        if vc is not None:
            self.vertex_colors = np.tile(np.array(vc), (len(self.vertices), 1))

    def rotate_vertices(self, rot_mat):
        if self.vertices is not None:
            self.vertices = (rot_mat @ self.vertices.T).T

    def translate(self, transl):
        if self.vertices is not None:
            self.vertices += transl

    def to_open3d(self):
        mesh = o3d.geometry.TriangleMesh()
        if self.vertices is not None:
            mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        if self.faces is not None:
            mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        if self.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        return mesh

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0

    ds_name = 'test'
    dataset_dir = '/data/fareal/EXTRACT/data/'
    mesh_base = '/data/fareal/EXTRACT/tools/object_meshes/contact_meshes/'

    ds_test = LoadData(dataset_dir=dataset_dir, ds_name=ds_name)
    
    data_loader = DataLoader(
        ds_test,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=int(arg.workers),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=grasp_data_collate
    )

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = DP(model).to(device=rank)
    model.eval()

    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")
    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    logger.info("Visualizing GrabNet's predictions - Press 'D' to show next sample")

    # 存储当前几何体的引用
    current_geometries = []

    with torch.no_grad():
        for intent in ['hold', 'use']:
            arg.intent = intent
            data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0} - Intent: {intent}")
            
            for i, inp in enumerate(data_iter):
                device = inp['verts_object'].device
                inp['obj_bps'] = inp['bps_object']
                inp['obj_verts_obj_ds'] = inp['verts_object']
                inp['hand_shape'] = torch.zeros(1, 10).to(device)

                if arg.intent is None:
                    prd, _ = model(inp=inp, step_idx=i, mode="test")
                else:
                    prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)

                h_faces = mano_layer.th_faces.detach().cpu().numpy()
                h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
                my_o_verts = inp['verts_object'].detach().cpu().numpy()

                for item_idx in range(h_verts_r.shape[0]):
                    show_next = False
                    
                    # 清除之前的几何体
                    for geom in current_geometries:
                        viz_ctx.remove_geometry(geom)
                    current_geometries = []

                    # 创建手部网格
                    hand_mesh = o3d.geometry.TriangleMesh()
                    hand_mesh.vertices = o3d.utility.Vector3dVector(h_verts_r[item_idx])
                    hand_mesh.triangles = o3d.utility.Vector3iVector(h_faces)
                    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(CMap["tomato"], (len(h_verts_r[item_idx]), 1)))

                    # 创建物体点云
                    obj_pcd = o3d.geometry.PointCloud()
                    obj_pcd.points = o3d.utility.Vector3dVector(my_o_verts[item_idx])
                    obj_pcd.colors = o3d.utility.Vector3dVector(np.tile(CMap["lime"], (len(my_o_verts[item_idx]), 1)))

                    # 添加新几何体并保存引用
                    viz_ctx.add_geometry(hand_mesh)
                    viz_ctx.add_geometry(obj_pcd)
                    current_geometries.extend([hand_mesh, obj_pcd])

                    # 等待用户按键
                    while not show_next:
                        viz_ctx.step()

if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent", type=str, default=None, choices=["use", "hold", "liftup"])
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)