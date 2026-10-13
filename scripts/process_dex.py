import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import trimesh
from joblib import Parallel, delayed
from tqdm import tqdm


sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')
from lib.utils.misc import RedirectStream


def extract_objects(args):
    """Extract object meshes from DexGraspNet"""
    dexgraspnet_dir = Path(args.dexgraspnet_data_dir)
    export_dir = Path(args.output_dir) / "original"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print("Scanning DexGraspNet dataset for objects...")
    obj_count = 0
    skipped_count = 0
    
    for obj_dir in tqdm(sorted(dexgraspnet_dir.iterdir()), desc="Extracting objects"):
        if not obj_dir.is_dir():
            continue
        
        obj_id = obj_dir.name
        
        # Check if already exported
        obj_path = export_dir / f"{obj_id}.obj"
        if obj_path.exists():
            skipped_count += 1
            continue
        
        # Find the decomposed.obj file
        coacd_dir = obj_dir / "coacd"
        if not coacd_dir.is_dir():
            continue
        
        decomposed_path = coacd_dir / "decomposed.obj"
        if not decomposed_path.exists():
            continue
        
        try:
            # Load and export
            mesh = trimesh.load(str(decomposed_path), process=False, skip_materials=True)
            mesh.export(str(obj_path), include_normals=False)
            obj_count += 1
        except Exception as e:
            print(f"Error exporting {obj_id}: {e}")
    
    print(f"Exported {obj_count} objects (skipped {skipped_count})")
    print(f"Output directory: {export_dir}")
    return obj_count


def watertight_one(oid, omesh, export_dir):
    """Make one mesh watertight using ManifoldPlus"""
    out_path = os.path.join(export_dir, f"{oid}.obj")
    if os.path.exists(out_path):
        return
    
    omesh.visual = trimesh.visual.ColorVisuals()
    temp_path = os.path.join(export_dir, f"_{oid}.obj")
    trimesh.exchange.export.export_mesh(omesh, temp_path, file_type="obj")

    try:
        depth_list = [8, 7, 6]
        for d in depth_list:
            command = [
                "thirdparty/ManifoldPlus/build/manifold", "--input", temp_path, "--output", out_path, "--depth",
                str(d)
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            obj_reload = trimesh.load(out_path, process=False, force="mesh", skip_materials=True)
            assert obj_reload.is_watertight, f"{out_path} is not watertight"
            if obj_reload.faces.shape[0] < 30000:
                break
        os.remove(temp_path)
    except Exception as e:
        with open(os.path.join(export_dir, f"{oid}.err"), "w") as f:
            f.write(str(e))


def watertight(args):
    """Make all objects watertight"""
    input_dir = os.path.join(args.output_dir, "original")
    export_dir = os.path.join(args.output_dir, "watertight")
    os.makedirs(export_dir, exist_ok=True)

    tasks = []
    skipped_count = 0
    for filename in tqdm(os.listdir(input_dir), desc="Preparing watertight tasks"):
        if filename.endswith(".obj"):
            oid = os.path.splitext(filename)[0]
            out_path = os.path.join(export_dir, f"{oid}.obj")
            if os.path.exists(out_path):
                skipped_count += 1
                continue
            omesh = trimesh.load(os.path.join(input_dir, filename), process=False, skip_materials=True)
            tasks.append(delayed(watertight_one)(oid, omesh, export_dir))

    print(f"Running watertight processing with {args.n_jobs} jobs (skipped {skipped_count})...")
    if tasks:
        Parallel(n_jobs=args.n_jobs, verbose=10, timeout=120)(tasks)
    else:
        print("All watertight files all done, nothing to do!")
    return


def voxel_one(oid, input_mesh_path, export_dir):
    """Voxelize one mesh using binvox"""
    out_mesh_dest = os.path.join(export_dir, f"{oid}.binvox")
    if os.path.exists(out_mesh_dest):
        return
    
    command = ["thirdparty/binvox", "-d", "128", input_mesh_path]
    subprocess.run(command)
    res_binvox_path = input_mesh_path.replace(".obj", ".binvox")
    if os.path.exists(res_binvox_path):
        shutil.move(res_binvox_path, out_mesh_dest)


def voxel(args):
    """Voxelize all watertight objects"""
    input_dir = os.path.join(args.output_dir, "watertight")
    export_dir = os.path.join(args.output_dir, "voxel")
    os.makedirs(export_dir, exist_ok=True)

    tasks = []
    skipped_count = 0
    all_wt_obj_flist = os.listdir(input_dir)
    for wt_obj_f in tqdm(all_wt_obj_flist, desc="Preparing voxel tasks"):
        if wt_obj_f.endswith(".obj"):
            oid = os.path.splitext(wt_obj_f)[0]
            out_mesh_dest = os.path.join(export_dir, f"{oid}.binvox")
            if os.path.exists(out_mesh_dest):
                skipped_count += 1
                continue
            input_mesh_path = os.path.join(input_dir, wt_obj_f)
            tasks.append(delayed(voxel_one)(oid, input_mesh_path, export_dir))

    print(f"Running voxelization with {args.n_jobs} jobs (skipped {skipped_count})...")
    if tasks:
        Parallel(n_jobs=args.n_jobs, verbose=10, timeout=120)(tasks)
    else:
        print("All voxel files already done, nothing to do!")
    return


def vhacd_one(oid, input_mesh_path, export_dir):
    """Decompose one mesh using V-HACD"""
    res_vhacd_path = os.path.join(export_dir, f"{oid}.obj")
    if os.path.exists(res_vhacd_path):
        return
    
    with RedirectStream(stream=sys.stdout), RedirectStream(stream=sys.stderr), io.StringIO() as \
        fstdout, contextlib.redirect_stdout(fstdout):
        import pybullet as pbl
        pbl.vhacd(
            fileNameIn=input_mesh_path,
            fileNameOut=res_vhacd_path,
            fileNameLogging="/dev/null",
            resolution=100000,
            concavity=0.0025,
            planeDownsampling=4,
            convexhullDownsampling=4,
            alpha=0.05,
            beta=0.0,
            pca=0,
            mode=0,
            maxNumVerticesPerCH=64,
            minVolumePerCH=0.0001,
        )


def vhacd(args):
    """Decompose all watertight objects using V-HACD"""
    input_dir = os.path.join(args.output_dir, "watertight")
    export_dir = os.path.join(args.output_dir, "vhacd")
    os.makedirs(export_dir, exist_ok=True)

    tasks = []
    skipped_count = 0
    all_wt_obj_flist = os.listdir(input_dir)
    for wt_obj_f in tqdm(all_wt_obj_flist, desc="Preparing V-HACD tasks"):
        if wt_obj_f.endswith(".obj"):
            oid = os.path.splitext(wt_obj_f)[0]
            res_vhacd_path = os.path.join(export_dir, f"{oid}.obj")
            if os.path.exists(res_vhacd_path):
                skipped_count += 1
                continue
            input_mesh_path = os.path.join(input_dir, wt_obj_f)
            tasks.append(delayed(vhacd_one)(oid, input_mesh_path, export_dir))

    print(f"Running V-HACD with {args.n_jobs} jobs (skipped {skipped_count})...")
    if tasks:
        Parallel(n_jobs=args.n_jobs, verbose=10, timeout=120)(tasks)
    else:
        print("All vhacd files already done, nothing to do!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DexGraspNet objects for physics evaluation")
    parser.add_argument("--dexgraspnet_data_dir", type=str, 
                       default="/home/zhenhua/grasp/DexGraspNet-main/meshdata",
                       help='Path to DexGraspNet meshdata directory')
    parser.add_argument("--output_dir", type=str, 
                       default="/home/zhenhua/grasp/DexGraspNet-main/post_process/",
                       help='Output directory for processed objects')
    parser.add_argument("--stage", type=str, default="all", 
                       choices=["extract", "watertight", "voxel", "vhacd", "all"],
                       help='Processing stage')
    parser.add_argument("--n_jobs", type=int, default=8, help='Number of parallel jobs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "extract" or args.stage == "all":
        print("=" * 60)
        print("Stage 1: Extracting objects from DexGraspNet dataset")
        print("=" * 60)
        extract_objects(args)
    
    if args.stage == "watertight" or args.stage == "all":
        print("=" * 60)
        print("Stage 2: Making objects watertight")
        print("=" * 60)
        watertight(args)
    
    if args.stage == "voxel" or args.stage == "all":
        print("=" * 60)
        print("Stage 3: Voxelize objects")
        print("=" * 60)
        voxel(args)
    
    if args.stage == "vhacd" or args.stage == "all":
        print("=" * 60)
        print("Stage 4: V-HACD decomposition")
        print("=" * 60)
        vhacd(args)
    
    print("=" * 60)
    print("Processing complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
