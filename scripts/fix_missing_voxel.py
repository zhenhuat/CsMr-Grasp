#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

proc_dir = "/home/zhenhua/grasp/AffordPose-main/postprocess_object"
watertight_dir = os.path.join(proc_dir, "watertight")
voxel_dir = os.path.join(proc_dir, "voxel")

# 缺少voxel的物体列表
missing_voxel_list = [
    'earphone_9657', 'faucet_1288', 'pot_4131', 'dispenser_4068', 
    'mug_8659', 'earphone_9913', 'jar_4233', 'earphone_9927', 
    'knife_1070', 'mug_8608', 'pot_5644', 'faucet_148', 'mug_8629'
]

def voxel_one(oid, input_mesh_path, export_dir):
    """Voxelize one mesh using binvox"""
    command = ["thirdparty/binvox", "-d", "128", input_mesh_path]
    subprocess.run(command, check=True, capture_output=True)
    res_binvox_path = input_mesh_path.replace(".obj", ".binvox")
    
    if not os.path.exists(res_binvox_path):
        print(f"ERROR: File {res_binvox_path} does not exist!")
        return False
    else:
        out_mesh_dest = os.path.join(export_dir, f"{oid}.binvox")
        shutil.move(res_binvox_path, out_mesh_dest)
        print(f"Created voxel for {oid}")
        return True

print(f"Processing {len(missing_voxel_list)} missing voxel files...")
success_count = 0
for oid in tqdm(missing_voxel_list, desc="Voxelizing"):
    input_mesh_path = os.path.join(watertight_dir, f"{oid}.obj")
    if os.path.exists(input_mesh_path):
        try:
            result = voxel_one(oid, input_mesh_path, voxel_dir)
            if result:
                success_count += 1
        except Exception as e:
            print(f"ERROR processing {oid}: {e}")

print(f"Done! Successfully processed {success_count}/{len(missing_voxel_list)} objects.")
