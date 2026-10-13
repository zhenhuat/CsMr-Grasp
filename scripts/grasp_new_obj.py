import argparse
import json
import os
import time
from argparse import Namespace

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ready_arguments
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.nn.parallel import DataParallel as DP
from trimesh import Trimesh

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

from lib.datasets.grasp_query import Queries
from lib.models import *
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.logger import logger
from lib.viztools.utils import ColorsMap as CMap
from lib.viztools.viz_o3d_utils import VizContext
from oikit.oi_shape.utils import ALL_INTENT

GrabNetConfig = dict(
    DATA_PRESET=dict(
        CENTER_IDX=9,
        N_RESAMPLED_OBJ_POINTS=4096,
    ),
    MODEL=dict(
        TYPE='GrabNet',
        COARSE_NET=dict(
            TYPE='CoarseNet',
            LATENTD=16,
            KL_COEF=0.005,
            VPE_PATH='assets/GrabNet/verts_per_edge.npy',
            C_WEIGHT_PATH='assets/GrabNet/rhand_weight.npy',
            PRETRAINED='checkpoints/grabnet_oishape/CoarseNet.pth.tar',
        ),
        REFINE_NET=dict(
            TYPE='RefineNet',
            KL_COEF=0.005,
            VPE_PATH='assets/GrabNet/verts_per_edge.npy',
            C_WEIGHT_PATH='assets/GrabNet/rhand_weight.npy',
            PRETRAINED='checkpoints/grabnet_oishape/RefineNet.pth.tar',
        ),
    ),
    TRANSFORM=dict(
        TYPE="GrabNetTransformObject",
        RAND_ROT=False,
        USE_ORIGINAL_OBJ_ROT=True,
        BPS_BASIS_PATH="assets/GrabNet/bps.npz",
        BPS_FEAT_TYPE="dists",
    ),
)


def load_obj_models(obj_path: str, n_sample_verts=10000, rescale=False):
    """
    Load object model and sample points from surface
    
    Args:
        obj_path: path to the OBJ/PLY file or JSON file containing mesh data
        n_sample_verts: number of points to sample from object surface (default: 10000)
        rescale: whether to rescale the object to fit in radius=0.1m sphere
    """
    if obj_path.endswith('.json'):
        # Load from JSON file (AffordPose format)
        with open(obj_path, 'r') as f:
            data = json.load(f)
        
        obj_mesh_data = data.get('object_mesh', [])
        
        # Parse vertices and faces from OBJ format strings
        vertices = []
        faces = []
        
        for item in obj_mesh_data:
            if isinstance(item, str):
                lines = item.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()[1:4]
                        if len(parts) >= 3:
                            vertex = [float(v) for v in parts]
                            vertices.append(vertex)
                    elif line.startswith('f '):
                        parts = line.split()[1:4]
                        if len(parts) >= 3:
                            face = [int(p.split('/')[0]) - 1 for p in parts]
                            faces.append(face)
        
        obj_verts = np.array(vertices, dtype=np.float32)
        obj_faces = np.array(faces, dtype=np.int32)
        
        if len(obj_faces) == 0 and len(obj_verts) > 2:
            # Fallback: simple triangulation if no faces found
            for i in range(0, len(obj_verts) - 2, 3):
                faces.append([i, i+1, i+2])
            obj_faces = np.array(faces, dtype=np.int32)
        
        maximum = obj_verts.max(0, keepdims=True)
        minimum = obj_verts.min(0, keepdims=True)
        offset = (maximum + minimum) / 2
        obj_verts = obj_verts - offset

        if rescale:
            scale = (obj_verts.max() - obj_verts.min()) / 2
            obj_verts = obj_verts / scale
            obj_verts = obj_verts * 0.1

        obj_rotmat = np.eye(3, dtype=np.float32)
    else:
        # Load from OBJ/PLY file
        obj_trimesh: Trimesh = trimesh.load(obj_path, process=False)
        obj_verts = np.asarray(obj_trimesh.vertices, dtype=np.float32)
        obj_faces = np.asarray(obj_trimesh.faces, dtype=np.int32)

        maximum = obj_verts.max(0, keepdims=True)
        minimum = obj_verts.min(0, keepdims=True)
        offset = (maximum + minimum) / 2
        obj_verts = obj_verts - offset

        if rescale:
            scale = (obj_verts.max() - obj_verts.min()) / 2
            obj_verts = obj_verts / scale
            obj_verts = obj_verts * 0.1

        obj_rotmat = np.eye(3, dtype=np.float32)

    # Sample points from mesh surface
    mesh = Meshes(verts=torch.from_numpy(obj_verts).unsqueeze(0).float(), faces=torch.from_numpy(obj_faces).unsqueeze(0))
    obj_verts_ds, obj_normals_ds = sample_points_from_meshes(mesh, n_sample_verts, return_normals=True)

    res = {
        Queries.SAMPLE_IDENTIFIER: obj_path,
        Queries.OBJ_ID: obj_path,
        Queries.OBJ_VERTS_OBJ: torch.from_numpy(obj_verts),
        Queries.OBJ_FACES: torch.from_numpy(obj_faces),
        Queries.OBJ_ROTMAT: torch.from_numpy(obj_rotmat),
        Queries.OBJ_VERTS_OBJ_DS: obj_verts_ds.squeeze(0),
        Queries.OBJ_NORMALS_OBJ_DS: obj_normals_ds.squeeze(0),
        # Add dummy hand data for transform (will be ignored during generation)
        Queries.HAND_POSE_OBJ: torch.zeros(48, dtype=torch.float32),
        Queries.HAND_SHAPE: torch.zeros(10, dtype=torch.float32),
        Queries.JOINTS_OBJ: torch.zeros((21, 3), dtype=torch.float32),
        Queries.HAND_VERTS_OBJ: torch.zeros((778, 3), dtype=torch.float32),
        Queries.HAND_TRANSL_OBJ: torch.zeros(3, dtype=torch.float32),
        # Add intent_id (default to 1 for "use")
        Queries.INTENT_ID: torch.tensor([1], dtype=torch.long),
    }

    return res


def grasp_new_obj(arg: Namespace, exp_time):
    rank = 0
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET)
    transform = builder.build_transform(cfg.TRANSFORM, data_preset=cfg.DATA_PRESET)

    model = DP(model).to(device=rank)
    transform = DP(transform).to(device=rank)
    model.eval()

    obj_data = load_obj_models(arg.obj_path, rescale=arg.rescale)
    for k, v in obj_data.items():
        if isinstance(v, torch.Tensor):
            obj_data[k] = v.unsqueeze(0).to(rank)
        if isinstance(v, np.ndarray):
            obj_data[k] = torch.from_numpy(v).unsqueeze(0).to(rank)

    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    show_next = False

    def next_sample(_):
        nonlocal show_next
        show_next = True

    viz_ctx.register_key_callback("D", next_sample)
    mano_rhand_path = os.path.join(arg.mano_path, "models", "MANO_RIGHT.pkl")
    mano_data = ready_arguments(mano_rhand_path)
    hand_faces = np.array(mano_data["f"]).astype(np.int32)

    res_hand_verts = []
    print("Press D to show next")
    for i in range(arg.n_grasps):
        show_next = False
        obj_data = transform(obj_data)
        
        if arg.intent is not None:
            intent_id = int(ALL_INTENT[arg.intent])
            intent_vec = torch.zeros((1, cfg.DATA_PRESET.N_INTENTS))
            intent_vec[0, intent_id - 1] = 1.0
            obj_data['intent_vec'] = intent_vec.to(rank)
            prd, _ = model(inp=obj_data, step_idx=0, mode="test", intent_name=arg.intent)
        else:
            prd, _ = model(inp=obj_data, step_idx=0, mode="test")

        hand_verts = prd["Refine.hand_verts"][0].detach().cpu().numpy()
        res_hand_verts.append(hand_verts)
        obj_verts = obj_data[Queries.OBJ_VERTS_OBJ][0].detach().cpu().numpy()
        obj_faces = obj_data[Queries.OBJ_FACES][0].detach().cpu().numpy()

        viz_ctx.update_by_mesh(f"hand", hand_verts, hand_faces, vcolors=CMap["tomato"])
        viz_ctx.update_by_mesh("obj", obj_verts, obj_faces, vcolors=CMap["lime"])

        while not show_next:
            viz_ctx.step()

    viz_ctx.deinit()

    if arg.save:
        ts = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(exp_time))
        save_path = os.path.join("user", "grasps", ts)
        os.makedirs(save_path, exist_ok=True)
        for i, hand_verts in enumerate(res_hand_verts):
            hand_mesh = Trimesh(vertices=hand_verts, faces=hand_faces)
            hand_mesh.export(os.path.join(save_path, f"hand_{i}.obj"))
        obj_mesh = Trimesh(vertices=obj_verts, faces=obj_faces)
        obj_mesh.export(os.path.join(save_path, "object.obj"))


if __name__ == '__main__':
    exp_time = time.time()
    arg, _ = parse_exp_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()
    logger.info(f"Using {world_size} GPUS")

    parser = argparse.ArgumentParser(description='extra')
    parser.add_argument("--obj_path", type=str, required=True, help='The path to the 3D object Mesh')
    parser.add_argument("--mano_path", type=str, default="assets/mano_v1_2", help='The path to MANO models')
    parser.add_argument("--n_grasps", type=int, required=False, default=1, help='how many grasps to generate')
    parser.add_argument("--intent",
                        type=str,
                        default=None,
                        choices=["use", "hold"],
                        help="specify intent for conditional generation")
    parser.add_argument("--rescale",
                        action="store_true",
                        default=False,
                        help='rescale the object to fit in radius=0.1m sphere')
    parser.add_argument("--save", action="store_true", default=False, help='save the grasps to file')
    arg_extra, _ = parser.parse_known_args()
    arg = argparse.Namespace(**vars(arg), **vars(arg_extra))

    grasp_new_obj(arg, exp_time)
