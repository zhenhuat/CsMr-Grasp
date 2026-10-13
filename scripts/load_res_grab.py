import argparse
import os
from argparse import Namespace
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/robin/fareal/icassp_grasp/')

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
from lib.viztools.utils import ColorsMap as CMap

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb

from pathlib import Path

# 创建保存路径
save_dir = Path("./mymesh/CsMr_grab/hold/")
save_dir.mkdir(parents=True, exist_ok=True)

# 颜色定义
COLOR_HAND = [0.82, 0.71, 0.55]  # 米黄色
COLOR_OBJECT = [1.0, 1.0, 0.0]   # 黄色

def is_valid_mesh(vertices, faces=None):
    """检查网格数据是否有效"""
    if vertices is None or len(vertices) == 0:
        return False
    if np.isnan(vertices).any():
        return False
    if faces is not None:
        if len(faces) == 0:
            return False
        if np.isnan(faces).any():
            return False
        if (faces < 0).any() or (faces >= len(vertices)).any():
            return False
    return True

def save_combined_mesh(hand_verts, hand_faces, obj_verts, obj_faces, save_path):
    """保存合并的手和物体网格"""
    try:
        # 创建手部网格
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
        hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
        hand_mesh.paint_uniform_color(COLOR_HAND)
        
        # 创建物体网格
        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
        obj_mesh.paint_uniform_color(COLOR_OBJECT)
        
        # 合并网格
        combined_mesh = hand_mesh + obj_mesh
        
        # 确保法线正确
        combined_mesh.compute_vertex_normals()
        
        # 保存为OBJ
        o3d.io.write_triangle_mesh(
            str(save_path), 
            combined_mesh, 
            write_vertex_normals=True,
            write_vertex_colors=True
        )
        return True
    except Exception as e:
        logger.error(f"保存网格失败: {e}")
        return False

class LoadData(Dataset):
    def __init__(self, dataset_dir, ds_name='train', dtype=torch.float32, only_params=False, load_on_ram=False, 
                 subjects=None, object_types=None, object_instances=None, object_actions=None):
        super().__init__()

        self.only_params = only_params
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path, 'grabnet_%s.npz' % ds_name))

        frame_names = np.load(os.path.join(dataset_dir, ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2] for name in self.frame_names])
        self.obj_names = [name.split('_')[0] for name in self.frame_objs]  # 提取并保存物体的名字（类型）

        # 过滤逻辑
        if subjects is not None and 'all' not in subjects:
            mask = np.isin(self.frame_sbjs, subjects)
            self.frame_names = self.frame_names[mask]
            self.frame_sbjs = self.frame_sbjs[mask]
            self.frame_objs = self.frame_objs[mask]
            self.obj_names = [self.obj_names[i] for i in range(len(self.obj_names)) if mask[i]]
            self.ds = self._filter_ds(mask)

        if object_types is not None or object_instances is not None or object_actions is not None:
            obj_types = np.asarray([name.split('_')[0] for name in self.frame_objs])
            obj_instances = np.asarray([name.split('_')[1] for name in self.frame_objs])
            obj_actions = np.asarray([name.split('_')[2] for name in self.frame_objs])

            mask = np.ones(len(self.frame_objs), dtype=bool)
            if object_types is not None and 'all' not in object_types:
                mask &= np.isin(obj_types, object_types)
            if object_instances is not None and 'all' not in object_instances:
                mask &= np.isin(obj_instances, object_instances)
            if object_actions is not None and 'all' not in object_actions:
                mask &= np.isin(obj_actions, object_actions)

            self.frame_names = self.frame_names[mask]
            self.frame_sbjs = self.frame_sbjs[mask]
            self.frame_objs = self.frame_objs[mask]
            self.obj_names = [self.obj_names[i] for i in range(len(self.obj_names)) if mask[i]]
            self.ds = self._filter_ds(mask)

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
        data_torch = {k: torch.tensor(data[k]) for k in data.files}
        return data_torch

    def _filter_ds(self, mask):
        filtered_ds = {}
        for key in self.ds.keys():
            filtered_ds[key] = self.ds[key][mask]
        return filtered_ds

    def load_disk(self, idx):
        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = [self._np2torch(f) for f in frame_names]
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]

    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params and not self.load_on_ram:
            form_disk = self.load_disk(idx)
            data_out.update(form_disk)

        # 加载并变换网格数据
        objname = self.obj_names[idx]
        mesh_path = os.path.join('/data/fareal/EXTRACT/tools/object_meshes/contact_meshes/', objname + '.ply')
        try:
            obj_mesh = Mesh(filename=mesh_path, vc=name_to_rgb['yellow'])
            rot_mat = data_out['root_orient_obj_rotmat'].cpu().numpy().reshape(3, 3).T
            transl = data_out['trans_obj'].cpu().numpy()
            obj_mesh.rotate_vertices(rot_mat)
            obj_mesh.v += transl

            # 将网格数据存储为顶点和面信息
            data_out['mesh_vertices'] = torch.tensor(obj_mesh.v, dtype=torch.float32)
            data_out['mesh_faces'] = torch.tensor(obj_mesh.f.astype(np.int32), dtype=torch.int32)
        except Exception as e:
            print(f"Failed to load mesh for {objname}: {e}")
            # 提供空数据作为后备
            data_out['mesh_vertices'] = torch.zeros((0, 3), dtype=torch.float32)
            data_out['mesh_faces'] = torch.zeros((0, 3), dtype=torch.int32)

        return data_out

def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0

    ds_name = 'test'
    dataset_dir = '/data/fareal/EXTRACT/data/'
    mesh_base = '/data/fareal/EXTRACT/tools/object_meshes/contact_meshes/'

    ds_test = LoadData(
        dataset_dir=dataset_dir, 
        ds_name=ds_name, 
        subjects=['s1'], 
        object_types=['mug','camera','wineglass','toothpaste'], 
        object_instances=['pass'], 
        object_actions=['1']
    )

    data_loader = DataLoader(ds_test,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             collate_fn=grasp_data_collate)

    print('DataLoader created successfully!')

    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    transform = DP(transform).to(device=rank)

    model: GrabNet = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
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
    logger.info("You are viz GrabNet's prediciton")



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
        for intent in ['hold']:
            data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0} - Intent: {intent}")
            for i, inp in enumerate(data_iter):
                device = inp['verts_object'].device
                inp['obj_bps'] = inp['bps_object'] 
                inp['obj_verts_obj_ds'] = inp['verts_object']
                inp['hand_shape'] = torch.zeros(1, 10).to(device)
                inp['memory_bps'] = memory_bps
                inp['memory_hand'] = memory_hand
                inp['memory_intent'] = memory_id

                if arg.intent is None:
                    prd, _ = model(inp=inp, step_idx=i, mode="test")
                else:
                    prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)

                h_faces = mano_layer.th_faces.detach().cpu().numpy()
                h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
                my_o_verts = inp['verts_object'].detach().cpu().numpy()
                
                n_items = h_verts_r.shape[0]
                
                for j in range(n_items):
                    hand_v = h_verts_r[j]
                    obj_v = my_o_verts[j]
                    
                    # 获取网格数据
                    mesh_vertices = inp['mesh_vertices'][j].cpu().numpy()
                    mesh_faces = inp['mesh_faces'][j].cpu().numpy()
                    
                    # 验证数据有效性
                    if not is_valid_mesh(hand_v, h_faces) or not is_valid_mesh(mesh_vertices, mesh_faces):
                        logger.warning(f"跳过无效网格数据 - 批次 {i}, 索引 {j}")
                        continue
                    
                    # 确保面索引是三角形
                    if mesh_faces.shape[1] != 3:
                        logger.warning(f"物体面不是三角形 - 批次 {i}, 索引 {j}")
                        continue
                    
                    # 保存合并的网格
                    save_path = save_dir / f"sample_{i * n_items + j:04d}.obj"
                    success = save_combined_mesh(
                        hand_v, h_faces,
                        mesh_vertices, mesh_faces,
                        save_path
                    )
                    
                    if success:
                        logger.info(f"成功保存到 {save_path}")
                    else:
                        logger.error(f"保存失败 {save_path}")

if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent", type=str, default=None, choices=["use", "hold", "liftup"], help="specify when use IntGen model")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)