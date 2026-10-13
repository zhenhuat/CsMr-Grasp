import argparse
import os
from argparse import Namespace
from time import time

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

import lib.models
from lib.datasets.grasp_query import Queries
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger
from lib.utils.misc import bar_prefix
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.etqdm import etqdm

from manotorch.manolayer import ManoLayer
from lib.viztools.viz_o3d_utils import VizContext
import open3d as o3d


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


class DexGraspNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, max_samples=100, rescale=True):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.samples = []
        self.max_samples = max_samples
        self.rescale = rescale
        
        print("Scanning DexGraspNet dataset...")
        obj_dirs = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                obj_dirs.append(item)
        
        print(f"Found {len(obj_dirs)} object directories")
        
        if max_samples > 0:
            obj_dirs = obj_dirs[:max_samples]
        
        print(f"Processing first {len(obj_dirs)} directories...")
        
        valid_count = 0
        for i, obj_dir in enumerate(sorted(obj_dirs)):
            if i % 100 == 0:
                print(f"Processing directory {i}/{len(obj_dirs)}")
            
            obj_path = os.path.join(dataset_dir, obj_dir)
            if not os.path.isdir(obj_path):
                continue
            
            coacd_dir = os.path.join(obj_path, 'coacd')
            if not os.path.isdir(coacd_dir):
                continue
            
            obj_file = os.path.join(coacd_dir, 'decomposed.obj')
            if not os.path.exists(obj_file):
                continue
            
            if '-' in obj_dir:
                category = obj_dir.split('-')[1]
            else:
                category = 'unknown'
            
            sample_id = obj_dir
            self.samples.append({
                'sample_id': sample_id,
                'obj_path': obj_file,
                'category': category,
            })
            valid_count += 1
        
        print(f"Found {valid_count} valid samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        import trimesh
        mesh = trimesh.load(sample_info['obj_path'], process=False)
        obj_vertices = np.asarray(mesh.vertices, dtype=np.float32)
        obj_faces = np.asarray(mesh.faces, dtype=np.int32)
        
        # Center the object
        maximum = obj_vertices.max(0, keepdims=True)
        minimum = obj_vertices.min(0, keepdims=True)
        offset = (maximum + minimum) / 2
        obj_vertices = obj_vertices - offset
        
        # Rescale to fit in radius=0.1m sphere (same as grasp_new_obj.py)
        if self.rescale:
            scale = (obj_vertices.max() - obj_vertices.min()) / 2
            if scale > 0:
                obj_vertices = obj_vertices / scale
                obj_vertices = obj_vertices * 0.1
        
        data_out = {
            'sample_id': sample_info['sample_id'],
            'obj_vertices': torch.from_numpy(obj_vertices),
            'obj_faces': torch.from_numpy(obj_faces),
            'category': sample_info['category'],
        }
        
        return data_out


def dexgraspnet_collate(batch):
    sample_ids = [item['sample_id'] for item in batch]
    categories = [item['category'] for item in batch]
    
    obj_vertices_list = []
    obj_faces_list = []
    
    for item in batch:
        obj_vertices_list.append(item['obj_vertices'])
        obj_faces_list.append(item['obj_faces'])
    
    return {
        'sample_id': sample_ids,
        'category': categories,
        'obj_vertices': obj_vertices_list,
        'obj_faces': obj_faces_list,
    }


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    
    dataset_dir = arg.dexgraspnet_data_dir
    ds_test = DexGraspNetDataset(
        dataset_dir=dataset_dir,
        max_samples=arg.max_samples,
        rescale=True
    )
    
    data_loader = DataLoader(ds_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             collate_fn=dexgraspnet_collate)
    
    rank = 0
    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    
    transform = DP(transform).to(device=rank)
    model = DP(model).to(device=rank)
    logger.info("Using GPU for computation")
    
    memory_bps = np.load('./memory//memory_bps.npy')
    memory_hand = np.load('./memory//memory_hand.npy')
    memory_id = np.load('./memory//memory_id.npy')
    
    memory_bps = torch.from_numpy(memory_bps).to(rank)
    memory_hand = torch.from_numpy(memory_hand).to(rank)
    memory_id = torch.from_numpy(memory_id).to(rank)
    
    model.eval()
    
    mano_layer = ManoLayer(center_idx=cfg.DATA_PRESET.CENTER_IDX, mano_assets_root="assets/mano_v1_2")
    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    
    show_next = False
    current_geometries = []
    
    def next_sample(_):
        nonlocal show_next
        show_next = True
    
    viz_ctx.register_key_callback("N", next_sample)
    logger.info("Visualizing DexGraspNet Generation Results - Press 'N' for next sample")
    
    intent_id = 1 if arg.intent == 'use' else 2
    
    with torch.no_grad():
        data_iter = etqdm(data_loader, desc=f"{bar_prefix['test']} Visualizing DexGraspNet")
        
        for i, inp in enumerate(data_iter):
            obj_vertices_list = inp['obj_vertices']
            obj_faces_list = inp['obj_faces']
            
            max_verts = max(v.shape[0] for v in obj_vertices_list)
            max_faces = max(f.shape[0] for f in obj_faces_list)
            
            obj_verts_padded = torch.zeros(len(obj_vertices_list), max_verts, 3, dtype=torch.float32)
            obj_faces_padded = torch.zeros(len(obj_faces_list), max_faces, 3, dtype=torch.int32)
            
            for j, (verts, faces) in enumerate(zip(obj_vertices_list, obj_faces_list)):
                obj_verts_padded[j, :verts.shape[0]] = verts
                obj_faces_padded[j, :faces.shape[0]] = faces
            
            n_sample_verts = cfg.DATA_PRESET.N_RESAMPLED_OBJ_POINTS
            obj_verts_ds_list = []
            obj_normals_ds_list = []
            
            for j, (verts, faces) in enumerate(zip(obj_vertices_list, obj_faces_list)):
                import trimesh
                mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
                points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                tri_normals = mesh.face_normals[face_indices]
                
                obj_verts_ds = torch.from_numpy(points).float()
                obj_normals_ds = torch.from_numpy(tri_normals).float()
                
                obj_verts_ds_list.append(obj_verts_ds)
                obj_normals_ds_list.append(obj_normals_ds)
            
            obj_verts_ds_padded = torch.zeros(len(obj_verts_ds_list), n_sample_verts, 3, dtype=torch.float32)
            obj_normals_ds_padded = torch.zeros(len(obj_normals_ds_list), n_sample_verts, 3, dtype=torch.float32)
            
            for j, (verts_ds, normals_ds) in enumerate(zip(obj_verts_ds_list, obj_normals_ds_list)):
                obj_verts_ds_padded[j, :verts_ds.shape[0]] = verts_ds
                obj_normals_ds_padded[j, :normals_ds.shape[0]] = normals_ds
            
            batch_size = len(obj_vertices_list)
            dummy_hand_pose = torch.zeros(batch_size, 48, dtype=torch.float32).to(rank)
            dummy_hand_shape = torch.zeros(batch_size, 10, dtype=torch.float32).to(rank)
            dummy_joints = torch.zeros(batch_size, 21, 3, dtype=torch.float32).to(rank)
            dummy_hand_verts = torch.zeros(batch_size, 778, 3, dtype=torch.float32).to(rank)
            dummy_hand_transl = torch.zeros(batch_size, 3, dtype=torch.float32).to(rank)
            
            dummy_intent_id = torch.tensor([intent_id] * batch_size, dtype=torch.long).to(rank)
            
            transform_inp = {
                Queries.OBJ_VERTS_OBJ: obj_verts_padded.to(rank),
                Queries.OBJ_FACES: obj_faces_padded.to(rank),
                Queries.OBJ_ROTMAT: torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(rank),
                Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds_padded.to(rank),
                Queries.OBJ_NORMALS_OBJ_DS: obj_normals_ds_padded.to(rank),
                Queries.HAND_POSE_OBJ: dummy_hand_pose,
                Queries.HAND_SHAPE: dummy_hand_shape,
                Queries.JOINTS_OBJ: dummy_joints,
                Queries.HAND_VERTS_OBJ: dummy_hand_verts,
                Queries.HAND_TRANSL_OBJ: dummy_hand_transl,
                Queries.INTENT_ID: dummy_intent_id,
                'sample_id': inp['sample_id'],
            }
            
            transform_inp = transform(transform_inp)
            
            transform_inp['memory_bps'] = memory_bps
            transform_inp['memory_hand'] = memory_hand
            transform_inp['memory_intent'] = memory_id
            
            intent_vec = torch.zeros((batch_size, cfg.DATA_PRESET.N_INTENTS))
            intent_vec[:, intent_id - 1] = 1.0
            transform_inp['intent_vec'] = intent_vec.to(rank)
            transform_inp['intent_name'] = arg.intent
            
            prd, _ = model(inp=transform_inp, step_idx=i, mode="test", intent_name=arg.intent)
            
            h_faces = mano_layer.th_faces.detach().cpu().numpy()
            h_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
            
            # Get original object vertices and faces for mesh visualization
            obj_verts = obj_verts_padded.detach().cpu().numpy()
            obj_faces = obj_faces_padded.detach().cpu().numpy()
            
            sample_ids = inp['sample_id']
            for item_idx in range(h_verts_r.shape[0]):
                show_next = False
                
                for geom in current_geometries:
                    viz_ctx.remove_geometry(geom)
                current_geometries = []
                
                # Create hand mesh
                hand_mesh = o3d.geometry.TriangleMesh()
                hand_mesh.vertices = o3d.utility.Vector3dVector(h_verts_r[item_idx])
                hand_mesh.triangles = o3d.utility.Vector3iVector(h_faces)
                hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(CMap["tomato"], (len(h_verts_r[item_idx]), 1)))
                
                # Create object mesh (instead of point cloud)
                obj_mesh = o3d.geometry.TriangleMesh()
                obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts[item_idx])
                obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces[item_idx])
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(CMap["lime"], (len(obj_verts[item_idx]), 1)))
                
                sample_id = sample_ids[item_idx]
                logger.info(f"Showing sample: {sample_id}, Intent: {arg.intent}")
                
                viz_ctx.add_geometry(hand_mesh)
                viz_ctx.add_geometry(obj_mesh)
                current_geometries.extend([hand_mesh, obj_mesh])
                
                while not show_next:
                    viz_ctx.step()
    
    logger.info("Visualization complete!")


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")
    
    parser = argparse.ArgumentParser(description='Visualize DexGraspNet Generation Results')
    parser.add_argument("--max_samples", type=int, default=10,
                       help='Maximum samples to visualize')
    parser.add_argument("--dexgraspnet_data_dir", type=str,
                       default="/home/zhenhua/grasp/DexGraspNet-main/meshdata",
                       help='Path to DexGraspNet meshdata directory')
    parser.add_argument("--intent", type=str, default="use", choices=["use", "hold"],
                       help="specify intent for conditional generation")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))
    
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Visualizing DexGraspNet Generation Results <====")
    main(cfg, arg, exp_time)
