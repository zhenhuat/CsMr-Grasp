#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

proc_dir = "/home/zhenhua/grasp/AffordPose-main/postprocess_object"
original_dir = os.path.join(proc_dir, "original")
watertight_dir = os.path.join(proc_dir, "watertight")
voxel_dir = os.path.join(proc_dir, "voxel")
vhacd_dir = os.path.join(proc_dir, "vhacd")

# 获取所有在original目录里的文件
original_files = [f for f in os.listdir(original_dir) if f.endswith(".obj")]

missing_voxel = []
missing_vhacd = []
missing_both = []

for f in original_files:
    oid = os.path.splitext(f)[0]
    has_voxel = os.path.exists(os.path.join(voxel_dir, f"{oid}.binvox"))
    has_vhacd = os.path.exists(os.path.join(vhacd_dir, f"{oid}.obj"))
    if not has_voxel and not has_vhacd:
        missing_both.append(oid)
    elif not has_voxel:
        missing_voxel.append(oid)
    elif not has_vhacd:
        missing_vhacd.append(oid)

print(f"Total original objects: {len(original_files)}")
print(f"Missing both voxel and vhacd: {len(missing_both)}")
print(f"Missing voxel only: {len(missing_voxel)}")
print(f"Missing vhacd only: {len(missing_vhacd)}")
print(f"Missing voxel total: {len(missing_both + missing_voxel)}")
print(f"Missing vhacd total: {len(missing_both + missing_vhacd)}")

print(f"\nMissing both: {missing_both[:20]}")
print(f"\nMissing voxel: {missing_voxel[:20]}")
print(f"\nMissing vhacd: {missing_vhacd[:20]}")

# 保存到文件
import pickle
with open(os.path.join(proc_dir, "missing_files.pkl"), "wb") as f:
    pickle.dump({
        "missing_voxel": missing_voxel,
        "missing_vhacd": missing_vhacd,
        "missing_both": missing_both
    }, f)

print(f"\nSaved to {os.path.join(proc_dir, 'missing_files.pkl')}")
