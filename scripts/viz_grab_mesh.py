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

class LoadData(Dataset):
    def __init__(self, dataset_dir, ds_name='train', dtype=torch.float32, only_params=False, load_on_ram=False, subjects=None, object_types=None, object_instances=None, object_actions=None):
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

            # 将网格数据存储为顶点和面信息，并转换类型
            data_out['mesh_vertices'] = torch.tensor(obj_mesh.v, dtype=torch.float32)
            # data_out['mesh_faces'] = torch.tensor(obj_mesh.f, dtype=torch.int32)
            data_out['mesh_faces'] = torch.tensor(obj_mesh.f.astype(np.int32), dtype=torch.int32)
        except Exception as e:
            print(f"Failed to load mesh for {objname}: {e}")

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
        object_types=['toothpaste'], 
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

    o3d_mesh = o3d.geometry.TriangleMesh()
    viz_ctx.register_key_callback("D", next_sample)
    logger.info("You are viz GrabNet's prediciton")

    with torch.no_grad():

        for intent in ['hold']:
            data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Epoch {0} - Intent: {intent}")
            for i, inp in enumerate(data_iter):
                device = inp['verts_object'].device
                inp['obj_bps'] = inp['bps_object'] 
                inp['obj_verts_obj_ds']=inp['verts_object']
                inp['hand_shape']=  torch.zeros(1, 10).to(device)

                if arg.intent is None:
                    prd, _ = model(inp=inp, step_idx=i, mode="test")
                else:
                    prd, _ = model(inp=inp, step_idx=i, mode="test", intent_name=arg.intent)

                h_faces = mano_layer.th_faces.detach().cpu().numpy()
                h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
                h_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
                my_o_verts = inp['verts_object'].detach().cpu().numpy()
                o_verts = inp['verts_object'].detach().cpu().numpy()

                n_items = h_verts_c.shape[0]
                viz_ofs = np.array([0.6, 0.0, 0.0])
                for j in range(n_items):
                    show_next = False
                    viz_ctx.update_by_mesh("hand_coarse", h_verts_r[j], h_faces, vcolors=[0.82, 0.71, 0.55])
                    my_ov = my_o_verts[j]
                    ov = o_verts[j]

                    # viz_ctx.update_by_pc("obj_ofs", ov , ov, pcolors=CMap["lime"])

                    # 可视化加载的网格
                    mesh_vertices = inp['mesh_vertices'][j].cpu().numpy()
                    mesh_faces = inp['mesh_faces'][j].cpu().numpy()

                    

                    o3d_mesh.vertices.clear()
                    o3d_mesh.triangles.clear()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.0]] * len(mesh_vertices)))
                    o3d_mesh.compute_vertex_normals()  # 可选，如果你需要更精确的光照效果
                    o3d_mesh.compute_triangle_normals()
                    viz_ctx.add_geometry(o3d_mesh)
                    viz_ctx.step()

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
    parser.add_argument("--intent", type=str, default=None, choices=["use", "hold", "liftup"], help="specify when use IntGen model")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Testing on single GPU (DP) <====")
    main(cfg, arg, exp_time)