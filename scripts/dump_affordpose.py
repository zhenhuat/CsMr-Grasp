import argparse
import hashlib
import os
import pickle
from argparse import Namespace
from time import time

import json
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

from oikit.oi_shape.utils import ALL_INTENT


class AffordPoseDataset(torch.utils.data.Dataset):
    """Dataset for loading AffordPose JSON files"""
    
    def __init__(self, dataset_dir, categories=None, max_samples_per_object=10):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.samples = []
        self.max_samples_per_object = max_samples_per_object
        
        print("Scanning AffordPose dataset...")
        if categories is None:
            # Scan all categories
            categories = [d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d)) and not d.endswith('.7z')]
        
        for category in sorted(categories):
            category_dir = os.path.join(dataset_dir, category)
            if not os.path.isdir(category_dir):
                continue
            
            print(f"Processing category: {category}")
            for obj_instance in sorted(os.listdir(category_dir)):
                obj_instance_dir = os.path.join(category_dir, obj_instance)
                if not os.path.isdir(obj_instance_dir):
                    continue
                
                for action in sorted(os.listdir(obj_instance_dir)):
                    action_dir = os.path.join(obj_instance_dir, action)
                    if not os.path.isdir(action_dir):
                        continue
                    
                    # Get JSON files
                    json_files = sorted([f for f in os.listdir(action_dir) if f.endswith('.json')])
                    
                    # Sample up to max_samples_per_object
                    sample_indices = np.linspace(0, len(json_files)-1, 
                                               min(max_samples_per_object, len(json_files)), 
                                               dtype=int)
                    
                    for idx in sample_indices:
                        json_file = json_files[idx]
                        json_path = os.path.join(action_dir, json_file)
                        
                        sample_id = f"{category}_{obj_instance}_{action}_{os.path.splitext(json_file)[0]}"
                        self.samples.append({
                            'sample_id': sample_id,
                            'json_path': json_path,
                            'category': category,
                            'instance': obj_instance,
                            'action': action,
                        })
        
        print(f"Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load JSON file
        with open(sample_info['json_path'], 'r') as f:
            data = json.load(f)
        
        # Parse object mesh
        obj_mesh_data = data.get('object_mesh', [])
        obj_vertices, obj_faces = self._parse_obj_mesh(obj_mesh_data)
        
        # Get object transformation
        trans_obj = np.array(data.get('trans_obj', [0.0, 0.0, 0.0]), dtype=np.float32)
        quat_obj = np.array(data.get('quat_obj', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rot_mat = R.from_quat(quat_obj).as_matrix().astype(np.float32)
        
        # Apply transformation to object vertices
        obj_vertices = obj_vertices @ rot_mat.T + trans_obj
        
        # Create data dict
        data_out = {
            'sample_id': sample_info['sample_id'],
            'obj_vertices': torch.from_numpy(obj_vertices),
            'obj_faces': torch.from_numpy(obj_faces),
            'obj_rotmat': torch.from_numpy(rot_mat),
            'obj_trans': torch.from_numpy(trans_obj),
            'category': sample_info['category'],
            'instance': sample_info['instance'],
            'action': sample_info['action'],
        }
        
        return data_out
    
    def _parse_obj_mesh(self, obj_mesh_data):
        """Parse object mesh from JSON format"""
        vertices = []
        faces = []
        
        if isinstance(obj_mesh_data, str):
            lines = obj_mesh_data.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()[1:4]
                    if len(parts) >= 3:
                        vertices.append([float(v) for v in parts])
                elif line.startswith('f '):
                    parts = line.split()[1:4]
                    if len(parts) >= 3:
                        face = [int(p.split('/')[0]) - 1 for p in parts]
                        faces.append(face)
        elif isinstance(obj_mesh_data, list):
            for item in obj_mesh_data:
                if isinstance(item, str):
                    lines = item.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('v '):
                            parts = line.split()[1:4]
                            if len(parts) >= 3:
                                vertices.append([float(v) for v in parts])
                        elif line.startswith('f '):
                            parts = line.split()[1:4]
                            if len(parts) >= 3:
                                face = [int(p.split('/')[0]) - 1 for p in parts]
                                faces.append(face)
        
        # Fallback triangulation if no faces
        if len(faces) == 0 and len(vertices) > 2:
            for i in range(0, len(vertices) - 2, 3):
                faces.append([i, i+1, i+2])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def affordpose_collate(batch):
    """Collate function for AffordPose dataset"""
    sample_ids = [item['sample_id'] for item in batch]
    categories = [item['category'] for item in batch]
    instances = [item['instance'] for item in batch]
    actions = [item['action'] for item in batch]
    
    # Find max vertices
    max_verts = max(item['obj_vertices'].shape[0] for item in batch)
    
    # Pad vertices and faces
    obj_vertices_list = []
    obj_faces_list = []
    obj_rotmats = []
    obj_trans = []
    
    for item in batch:
        obj_vertices_list.append(item['obj_vertices'])
        obj_faces_list.append(item['obj_faces'])
        obj_rotmats.append(item['obj_rotmat'])
        obj_trans.append(item['obj_trans'])
    
    return {
        'sample_id': sample_ids,
        'category': categories,
        'instance': instances,
        'action': actions,
        'obj_vertices': obj_vertices_list,
        'obj_faces': obj_faces_list,
        'obj_rotmat': torch.stack(obj_rotmats),
        'obj_trans': torch.stack(obj_trans),
    }


def main(cfg: CN, arg: Namespace, time_f: float):
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    rank = 0
    
    # Create output directory
    dump_path = arg.output_dir
    os.makedirs(dump_path, exist_ok=True)
    logger.info(f"Dumping results to {dump_path}")
    
    def dumper(prd, inp, step_idx, **kwargs):
        """Save generated grasps to pickle files"""
        sample_ids = inp['sample_id']
        obj_rotmat = inp[Queries.OBJ_ROTMAT].detach().cpu().numpy()
        
        hand_verts_c = prd["Coarse.hand_verts"].detach().cpu().numpy()
        hand_verts_r = prd["Refine.hand_verts"].detach().cpu().numpy()
        
        n_samples = hand_verts_r.shape[0]
        
        for i in range(n_samples):
            sample_id = sample_ids[i]
            grasp_hash = hashlib.md5(hand_verts_r[i].tobytes()).hexdigest()[:10]
            
            sample_fname = f"{sample_id}_{grasp_hash}.pkl"
            
            # Get object vertices for this sample
            obj_verts_ds = inp[Queries.OBJ_VERTS_OBJ_DS][i].detach().cpu().numpy()
            
            sample_content = {
                "sample_id": sample_id,
                "obj_rotmat": obj_rotmat[i].reshape(3, 3),
                "obj_verts_ds": obj_verts_ds,
                "hand_verts_c": hand_verts_c[i],
                "hand_verts_r": hand_verts_r[i],
                "grasp_hash": grasp_hash,
            }
            
            with open(os.path.join(dump_path, sample_fname), "wb") as f:
                pickle.dump(sample_content, f)
        
        return True
    
    # Build dataset
    dataset_dir = arg.affordpose_data_dir
    categories = None
    if arg.categories is not None:
        categories = arg.categories.split(',')
    
    ds_test = AffordPoseDataset(
        dataset_dir=dataset_dir,
        categories=categories,
        max_samples_per_object=arg.max_samples
    )
    
    data_loader = DataLoader(ds_test,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn,
                             collate_fn=affordpose_collate)
    
    # Build transform and model
    transform = builder.build_transform(CN(cfg.TRANSFORM), data_preset=cfg.DATA_PRESET)
    
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    
    # Use GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        transform = DP(transform).to(device=rank)
        model = DP(model).to(device=rank)
        device = torch.device(f"cuda:{rank}")
    else:
        # Use CPU without DataParallel
        transform = transform.to(device='cpu')
        model = model.to(device='cpu')
        device = torch.device('cpu')
        logger.info("Using CPU for computation")

    memory_bps = np.load('./memory//memory_bps.npy')
    memory_hand = np.load('./memory//memory_hand.npy')
    memory_id = np.load('./memory//memory_id.npy')
    # memory_ov = np.load('./memory/memory_ov.npy')
    # memory_transl = np.load('./memory/memory_transl.npy')

    # 将NumPy数组转换为PyTorch Tensor
    memory_bps = torch.from_numpy(memory_bps)
    memory_hand = torch.from_numpy(memory_hand)
    memory_id = torch.from_numpy(memory_id)

    
    model.eval()
    
    # Generate grasps
    with torch.no_grad():
        data_iter = etqdm(data_loader, rank=rank, desc=f"{bar_prefix['test']} Generating grasps")
        
        for i, inp in enumerate(data_iter):
            # Prepare input data (device is already set above)
            
            # Process object vertices
            obj_vertices_list = inp['obj_vertices']
            obj_faces_list = inp['obj_faces']
            
            # Find max vertices for padding
            max_verts = max(v.shape[0] for v in obj_vertices_list)
            max_faces = max(f.shape[0] for f in obj_faces_list)
            
            # Pad and stack
            obj_verts_padded = torch.zeros(len(obj_vertices_list), max_verts, 3, dtype=torch.float32)
            obj_faces_padded = torch.zeros(len(obj_faces_list), max_faces, 3, dtype=torch.int32)
            
            for j, (verts, faces) in enumerate(zip(obj_vertices_list, obj_faces_list)):
                obj_verts_padded[j, :verts.shape[0]] = verts
                obj_faces_padded[j, :faces.shape[0]] = faces
            
            # Sample points from mesh surface using trimesh (alternative to pytorch3d)
            # We need to sample 4096 points with normals for each object in the batch
            n_sample_verts = cfg.DATA_PRESET.N_RESAMPLED_OBJ_POINTS  # 4096
            obj_verts_ds_list = []
            obj_normals_ds_list = []
            
            for j, (verts, faces) in enumerate(zip(obj_vertices_list, obj_faces_list)):
                # Create trimesh object
                import trimesh
                mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy(), process=False)
                
                # Sample points from surface (this returns points and their face indices)
                points, face_indices = mesh.sample(n_sample_verts, return_index=True)
                
                # Get normals at the sampled points by interpolating vertex normals
                # Get the vertices and normals of the sampled faces
                tri_vertices = mesh.triangles[face_indices]  # (n_points, 3, 3)
                tri_normals = mesh.face_normals[face_indices]  # (n_points, 3)
                
                # For simplicity, use face normals (this is what grabnet_transform expects)
                obj_verts_ds = torch.from_numpy(points).float()
                obj_normals_ds = torch.from_numpy(tri_normals).float()
                
                obj_verts_ds_list.append(obj_verts_ds)
                obj_normals_ds_list.append(obj_normals_ds)
            
            # Pad sampled vertices and normals
            obj_verts_ds_padded = torch.zeros(len(obj_verts_ds_list), n_sample_verts, 3, dtype=torch.float32)
            obj_normals_ds_padded = torch.zeros(len(obj_normals_ds_list), n_sample_verts, 3, dtype=torch.float32)
            
            for j, (verts_ds, normals_ds) in enumerate(zip(obj_verts_ds_list, obj_normals_ds_list)):
                obj_verts_ds_padded[j, :verts_ds.shape[0]] = verts_ds
                obj_normals_ds_padded[j, :normals_ds.shape[0]] = normals_ds
            
            # Create dummy hand data (will be ignored during generation, but required by transform)
            batch_size = len(obj_vertices_list)
            dummy_hand_pose = torch.zeros(batch_size, 48, dtype=torch.float32).to(device)
            dummy_hand_shape = torch.zeros(batch_size, 10, dtype=torch.float32).to(device)
            dummy_joints = torch.zeros(batch_size, 21, 3, dtype=torch.float32).to(device)
            dummy_hand_verts = torch.zeros(batch_size, 778, 3, dtype=torch.float32).to(device)
            dummy_hand_transl = torch.zeros(batch_size, 3, dtype=torch.float32).to(device)
            
            # Determine intent ID
            if arg.intent is not None:
                intent_id = int(ALL_INTENT[arg.intent])
            else:
                intent_id = 1  # default to "use"
            
            dummy_intent_id = torch.tensor([intent_id] * batch_size, dtype=torch.long).to(device)
            
            transform_inp = {
                Queries.OBJ_VERTS_OBJ: obj_verts_padded.to(device),
                Queries.OBJ_FACES: obj_faces_padded.to(device),
                Queries.OBJ_ROTMAT: inp['obj_rotmat'].to(device),
                Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds_padded.to(device),
                Queries.OBJ_NORMALS_OBJ_DS: obj_normals_ds_padded.to(device),
                Queries.HAND_POSE_OBJ: dummy_hand_pose,
                Queries.HAND_SHAPE: dummy_hand_shape,
                Queries.JOINTS_OBJ: dummy_joints,
                Queries.HAND_VERTS_OBJ: dummy_hand_verts,
                Queries.HAND_TRANSL_OBJ: dummy_hand_transl,
                Queries.INTENT_ID: dummy_intent_id,
                'sample_id': inp['sample_id'],
            }
            
            # Apply transform to get BPS features and other preprocessing
            transform_inp = transform(transform_inp)

            transform_inp['memory_bps'] = memory_bps
            transform_inp['memory_hand'] = memory_hand
            transform_inp['memory_intent'] = memory_id
            
            # Add intent vector if specified
            if arg.intent is not None:
                intent_id = int(ALL_INTENT[arg.intent])
                intent_vec = torch.zeros((batch_size, cfg.DATA_PRESET.N_INTENTS))
                intent_vec[:, intent_id - 1] = 1.0
                transform_inp['intent_vec'] = intent_vec.to(device)
                transform_inp['intent_name'] = arg.intent
            
            # Generate grasps
            prd, _ = model(inp=transform_inp, step_idx=i, mode="test",intent_name=arg.intent, callback=dumper)
    
    logger.info(f"Finished dumping results to {dump_path}")


if __name__ == '__main__':
    exp_time = time()
    arg, _ = parse_exp_args()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")
    
    parser = argparse.ArgumentParser(description='Dump AffordPose grasps')
    parser.add_argument("--output_dir", type=str,
                       default="/home/zhenhua/grasp/OakInk-Grasp-Generation-main/afford_result/csmr_hold",
                       help='Output directory for dumped grasps')
    parser.add_argument("--affordpose_data_dir", type=str,
                       default="/home/zhenhua/grasp/AffordPose-main/AffordPose Data/",
                       help='Path to AffordPose Data directory')
    parser.add_argument("--categories", type=str, default=None,
                       help='Comma-separated list of categories to process (default: all)')
    parser.add_argument("--max_samples", type=int, default=1,
                       help='Maximum samples per object instance')
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold"],
                        help="specify intent for conditional generation")
    arg_extra, _ = parser.parse_known_args(_)
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))
    
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    logger.info("====> Dumping AffordPose grasps <====")
    main(cfg, arg, exp_time)
