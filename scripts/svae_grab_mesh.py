import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
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
from lib.viztools.utils import ColorsMap as CMap


import numpy as np
import os
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb



import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
import open3d as o3d

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
import open3d as o3d

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
import open3d as o3d

def save_mesh_as_ply(vertices, faces, save_dir, objname, file_name):
    os.makedirs(save_dir, exist_ok=True)
    ply_filename = f"{objname}_{file_name}.ply"
    ply_filepath = os.path.join(save_dir, ply_filename)
    obj_filename = f"{objname}_{file_name}.obj"
    obj_filepath = os.path.join('/data/fareal/GRAB_PROCESS_EVA/', obj_filename)

    if not os.path.exists(ply_filepath):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(ply_filepath, o3d_mesh)
        
        # 转换为 OBJ 文件
    convert_ply_to_obj(ply_filepath, obj_filepath)
    
    return obj_filepath

def convert_ply_to_obj(ply_path, obj_path):
    # 读取 PLY 文件
    mesh = o3d.io.read_triangle_mesh(ply_path)
    # 保存为 OBJ 文件
    o3d.io.write_triangle_mesh(obj_path, mesh)

class SaveMeshData(Dataset):
    def __init__(self, dataset_dir, ds_name='train', save_dir='saved_meshes', record_file='mesh_paths.json', dtype=np.float32, only_params=False, subjects=None, object_types=None, object_instances=None, object_actions=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.ds_name = ds_name
        self.save_dir = save_dir
        self.record_file = record_file
        self.only_params = only_params
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path, 'grabnet_%s.npz' % ds_name))

        frame_names = np.load(os.path.join(dataset_dir, ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2] for name in self.frame_names])
        self.obj_names = [name.split('_')[0] for name in self.frame_objs]
        self.file_names = [os.path.splitext(os.path.basename(fname))[0] for fname in frame_names]

        # 过滤逻辑
        if subjects is not None and 'all' not in subjects:
            mask = np.isin(self.frame_sbjs, subjects)
            self.frame_names = self.frame_names[mask]
            self.frame_sbjs = self.frame_sbjs[mask]
            self.frame_objs = self.frame_objs[mask]
            self.obj_names = [self.obj_names[i] for i in range(len(self.obj_names)) if mask[i]]
            self.file_names = [self.file_names[i] for i in range(len(self.file_names)) if mask[i]]
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
            self.file_names = [self.file_names[i] for i in range(len(self.file_names)) if mask[i]]
            self.ds = self._filter_ds(mask)

    def _np2torch(self, ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k: torch.tensor(data[k], dtype=torch.float32) for k in data.files}
        return data_torch

    def _filter_ds(self, mask):
        filtered_ds = {}
        for key in self.ds.keys():
            filtered_ds[key] = self.ds[key][mask]
        return filtered_ds

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        objname = self.obj_names[idx]
        file_name = self.file_names[idx]
        mesh_path = os.path.join('/data/fareal/EXTRACT/tools/object_meshes/', 'contact_meshes', objname + '.ply')
        try:
            obj_mesh = Mesh(filename=mesh_path, vc=name_to_rgb['yellow'])
            rot_mat = data_out['root_orient_obj_rotmat'].numpy().reshape(3, 3).T
            transl = data_out['trans_obj'].numpy()
            obj_mesh.rotate_vertices(rot_mat)
            obj_mesh.v += transl

            mesh_vertices = obj_mesh.v
            mesh_faces = obj_mesh.f.astype(np.int32)
            ply_path = save_mesh_as_ply(mesh_vertices, mesh_faces, self.save_dir, objname, file_name)
            
            return ply_path
        except Exception as e:
            print(f"Failed to load mesh for {objname}: {e}")
            return None

    def save_all_meshes(self):
        mesh_paths = {}
        for idx in range(len(self)):
            ply_path = self[idx]
            if ply_path:
                objname = self.obj_names[idx]
                file_name = self.file_names[idx]
                mesh_paths[f"{objname}_{file_name}"] = ply_path

        with open(self.record_file, 'w') as f:
            json.dump(mesh_paths, f)



# 使用 SaveMeshData 类保存所有网格
dataset_dir = '/data/fareal/EXTRACT/data'  
save_dir = '/data/fareal/GRAB_PROCESS_EVA/'
record_file = '/data/fareal/GRAB_PROCESS_EVA/mesh_paths.json'
save_data = SaveMeshData(
    dataset_dir=dataset_dir,
    ds_name='test',
    save_dir=save_dir,
    record_file=record_file,
    subjects=['s1'], 
    object_types=['mug','camera','wineglass','toothpaste'], 
    object_instances=['pass'], 
    object_actions=['1']
)
save_data.save_all_meshes()


exit()

