import argparse
import os
import pickle

import numpy as np
import torch
import trimesh
from joblib import Parallel, delayed
from trimesh import Trimesh

import sys
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')

from lib.metrics.basic_metric import AverageMeter
from lib.metrics.disjointedness import region_disjointness_metric
from lib.metrics.intersection import solid_intersection_volume
from lib.metrics.penetration import penetration
from lib.metrics.simulator import simulation_sample


class DumpedDexGraspsLoader(object):

    def __init__(self, dumped_grasps_dir, dexgraspnet_mesh_dir):
        self.dumped_grasps_dir = dumped_grasps_dir
        self.dexgraspnet_mesh_dir = dexgraspnet_mesh_dir
        grasp_fname = sorted(os.listdir(dumped_grasps_dir))
        grasp_fname = [el for el in grasp_fname if el.endswith(".pkl")]
        self.grasp_files = [os.path.join(dumped_grasps_dir, el) for el in grasp_fname]
        
        hand_mesh_wt = trimesh.load("assets/hand_mesh_watertight.obj", process=False)
        self.hand_wt_faces = np.array(hand_mesh_wt.faces, dtype=np.int32)

    def __len__(self):
        return len(self.grasp_files)

    def __getitem__(self, idx):
        with open(self.grasp_files[idx], "rb") as f:
            grasp_item = pickle.load(f)

        sample_id = os.path.basename(self.grasp_files[idx]).split(".")[0]
        grasp_item["sample_id"] = sample_id
        
        # 从 sample_id 中提取 obj_id (sample_id 格式可能是: 
        # 1. core-{category}-{hash}_{grasp_hash}
        # 2. sem-{category}-{hash}_{grasp_hash}
        # 去掉 grasp_hash
        if '_' in sample_id:
            obj_id = sample_id.rsplit('_', 1)[0]  # 去掉 grasp_hash
        else:
            obj_id = sample_id
        
        # 如果 obj_id 包含完整路径，需要提取相对路径
        if '/post_process/' in obj_id or '/meshdata/' in obj_id:
            # 从完整路径中提取物体目录名 (如 core-bottle-xxx)
            obj_id = obj_id.split('/')[-1]
        
        # 构建物体文件路径 - 尝试多个可能的目录
        coacd_dir = os.path.join(self.dexgraspnet_mesh_dir, obj_id, 'coacd')
        decomposed_obj_path = os.path.join(coacd_dir, 'decomposed.obj')
        
        if not os.path.exists(decomposed_obj_path):
            raise FileNotFoundError(f"找不到物体文件: {decomposed_obj_path}")
        
        # 加载 watertight 物体网格
        obj_mesh = trimesh.load(decomposed_obj_path, process=False)
        obj_wt = obj_mesh
        
        # 查找 convex pieces 用于仿真
        convex_pieces = sorted([f for f in os.listdir(coacd_dir) 
                               if f.startswith('coacd_convex_piece_') and f.endswith('.obj')])
        
        # 使用第一个 convex piece 作为仿真文件（或者组合多个）
        if convex_pieces:
            vhacd_path = os.path.join(coacd_dir, convex_pieces[0])
        else:
            vhacd_path = decomposed_obj_path
        
        grasp_item["obj_wt"] = obj_wt
        grasp_item["obj_id"] = obj_id
        grasp_item["vhacd_path"] = vhacd_path
        grasp_item["hand_faces"] = self.hand_wt_faces
        return grasp_item


def evaluate_dex_grasps(idx, grasp_loader, palm_vert_assignment, palm_vert_idx, sims_dir):
    try:
        grasp_item = grasp_loader[idx]
        sample_id = grasp_item["sample_id"]
        obj_wt: Trimesh = grasp_item["obj_wt"]
        vhacd_path: str = grasp_item["vhacd_path"]
        obj_rotmat = grasp_item["obj_rotmat"]
        hand_verts = grasp_item["hand_verts_r"]
        hand_faces = grasp_item["hand_faces"]

        obj_wt_verts = np.asarray(obj_wt.vertices, dtype=np.float32)
        obj_wt_faces = np.array(obj_wt.faces, dtype=np.int32)

        # penetration
        pentr_dep = penetration(obj_verts=obj_wt_verts, obj_faces=obj_wt_faces, hand_verts=hand_verts)

        # solid intersection volume - 使用简化的体素化方法
        try:
            # 创建体素化网格用于体积计算
            vox_size = 0.005  # 5mm voxel size
            
            # 检查是否有有效顶点
            if len(obj_wt_verts) == 0:
                raise ValueError("Object has no vertices")
            
            # 简化的体素化方法 - 使用 trimesh 自带的 voxelize
            obj_vox = obj_wt.copy()
            obj_vox = obj_vox.apply_translation(-obj_vox.centroid)
            
            # 创建体素网格
            pitch = vox_size
            bounding_box = obj_vox.bounding_box
            bounds = bounding_box.bounds
            
            # 计算体素网格的形状
            sizes = bounds[1] - bounds[0]
            grid_shape = np.ceil(sizes / pitch).astype(int) + 2
            
            # 创建密集体素网格
            vox_grid = np.zeros(grid_shape, dtype=bool)
            
            # 手动进行体素化 - 检查每个顶点落在哪个体素中
            verts_centered = obj_wt_verts - obj_wt_verts.mean(axis=0)
            valid_mask = np.all(np.abs(verts_centered) < sizes.max() / 2 + pitch, axis=1)
            valid_verts = verts_centered[valid_mask]
            
            if len(valid_verts) > 0:
                # 计算体素索引
                voxel_indices = np.floor((valid_verts + sizes.max() / 2 + pitch) / pitch).astype(int)
                valid_voxel_mask = np.all(voxel_indices >= 0, axis=1) & np.all(voxel_indices < grid_shape, axis=1)
                voxel_indices = voxel_indices[valid_voxel_mask]
                
                # 标记被占用的体素
                for idx_tuple in voxel_indices:
                    vox_grid[tuple(idx_tuple)] = True
                
                # 计算体素体积
                obj_element_volume = pitch ** 3
                
                # 获取所有被占用的体素中心点
                occupied_voxel_indices = np.argwhere(vox_grid)
                obj_vox_points = occupied_voxel_indices * pitch + bounds[0]
                
                pentr_vol = solid_intersection_volume(hand_verts=hand_verts,
                                                      hand_faces=hand_faces,
                                                      obj_vox_points=obj_vox_points,
                                                      obj_vox_el_vol=obj_element_volume,
                                                      return_kin=False)
            else:
                pentr_vol = 0.0
                
        except Exception as e:
            print(f"Warning: Volume calculation failed for {sample_id}: {e}")
            pentr_vol = 0.0

        # disjoint distance
        _, disjo_dist = region_disjointness_metric(hand_verts=hand_verts,
                                                   obj_verts=obj_wt_verts,
                                                   hand_region_assignment=palm_vert_assignment)

        # simulation displacement - 使用 convex pieces 进行仿真
        try:
            sims_disp = simulation_sample(sample_idx=idx,
                                          sample_info={
                                              "sample_id": sample_id,
                                              "hand_verts": hand_verts,
                                              "hand_faces": hand_faces,
                                              "obj_verts": obj_wt_verts,
                                              "obj_faces": obj_wt_faces,
                                              "obj_vhacd_fname": vhacd_path,
                                              "obj_rotmat": obj_rotmat
                                          },
                                          save_gif_folder=os.path.join(sims_dir, "gif"),
                                          save_obj_folder=os.path.join(sims_dir, "vhacd"),
                                          tmp_folder=os.path.join(sims_dir, "tmp"),
                                          use_gui=False,
                                          sample_vis_freq=1)
        except Exception as e:
            print(f"Warning: Simulation failed for {sample_id}: {e}")
            sims_disp = -1.0

        eval_res = {
            "sample_id": sample_id,
            "pentr_dep": pentr_dep,
            "pentr_vol": pentr_vol,
            "disjo_dist": disjo_dist,
            "sims_disp": sims_disp,
        }
        return eval_res
    except Exception as e:
        print(f"Sample {idx} 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(arg):
    dumped_grasps_dir = arg.dumped_grasps_dir
    dexgraspnet_mesh_dir = arg.dexgraspnet_mesh_dir
    simulation_dir = os.path.join(arg.exp_path, "simulation")
    evaluation_dir = os.path.join(arg.exp_path, "evaluations")
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    dumped_grasps_loader = DumpedDexGraspsLoader(dumped_grasps_dir, dexgraspnet_mesh_dir)
    
    # 加载手部区域分配
    try:
        from manotorch.utils.anchorutils import anchor_load_driver
        merged_vertex_assignment = anchor_load_driver("./assets")[2]
    except:
        print("Warning: Could not load anchor driver, using default assignment")
        merged_vertex_assignment = None
    
    hand_palm_vert_idx = np.loadtxt("assets/hand_palm_full.txt", dtype=np.int32)

    task_list = []
    print(f"Total number of grasps: {len(dumped_grasps_loader)} to be evaluated.")
    for i in range(len(dumped_grasps_loader)):
        task_list.append(
            delayed(evaluate_dex_grasps)(i, dumped_grasps_loader, merged_vertex_assignment, hand_palm_vert_idx,
                                     simulation_dir))

    pentr_dep = AverageMeter("pentr_dep")
    pentr_vol = AverageMeter("pentr_vol")
    disjo_dist = AverageMeter("disjo_dist")
    sims_disp = AverageMeter("sims_disp")
    sims_disp_list = []  # for calculating std

    pentr_dep_list = []
    pentr_vol_list = []

    eval_res = Parallel(n_jobs=arg.n_jobs, verbose=10)(task_list)
    
    # dump eval_res
    with open(os.path.join(evaluation_dir, "eval_res.pkl"), "wb") as f:
        pickle.dump(eval_res, f)

    # 过滤掉失败的样本（None）
    valid_res = [res for res in eval_res if res is not None]
    print(f"Valid results: {len(valid_res)} / {len(eval_res)}")

    for i, sample_res in enumerate(valid_res):
        pentr_dep.update(sample_res["pentr_dep"])
        pentr_vol.update(sample_res["pentr_vol"])
        disjo_dist.update(sample_res["disjo_dist"])
        if sample_res["sims_disp"] >= 0:  # 只统计有效值
            sims_disp.update(sample_res["sims_disp"])
            sims_disp_list.append(sample_res["sims_disp"])

        pentr_dep_list.append(sample_res["pentr_dep"])
        pentr_vol_list.append(sample_res["pentr_vol"])

    with open(os.path.join(evaluation_dir, "Metric.txt"), "a") as f:
        f.write(f"Valid samples: {len(valid_res)} / {len(eval_res)}\n\n")
        
        if len(valid_res) > 0:
            f.write(f"pentr_dep mean: {pentr_dep.avg}\n")
            f.write(f"pentr_dep std: {np.std(pentr_dep_list)}\n")

            f.write(f"pentr_vol mean: {pentr_vol.avg}\n")
            f.write(f"pentr_vol std: {np.std(pentr_vol_list)}\n")

            f.write(f"disjo_dist mean: {disjo_dist.avg}\n")

            f.write(f"sims_disp mean: {sims_disp.avg}\n")
            f.write(f"sims_disp std: {np.std(sims_disp_list)}\n")

    print(f"Evaluation done!, results are saved in: {evaluation_dir}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval_dump for DexGraspNet')
    parser.add_argument("-g", "--gpu_id", type=str, default=0, help="override enviroment var CUDA_VISIBLE_DEVICES")
    parser.add_argument("--n_jobs", type=int, default=8, help="number of jobs for parallel processing")
    parser.add_argument("--dumped_grasps_dir", type=str, required=True, help="directory containing dumped grasp pkl files")
    parser.add_argument("--exp_path", type=str, required=True, help="path to save experiment results")
    parser.add_argument("--dexgraspnet_mesh_dir", type=str,
                       default="/home/zhenhua/grasp/DexGraspNet-main/meshdata",
                       help="directory containing DexGraspNet mesh files (coacd/decomposed.obj)")
    arg = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    world_size = torch.cuda.device_count()

    main(arg)
