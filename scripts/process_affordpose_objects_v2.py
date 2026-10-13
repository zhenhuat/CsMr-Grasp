import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import trimesh
from joblib import Parallel, delayed
from tqdm import tqdm


# Import from OakInk
sys.path.append('/home/zhenhua/grasp/OakInk-Grasp-Generation-main/')
from lib.utils.misc import RedirectStream


def parse_obj_mesh(obj_mesh_data):
    """Parse object mesh data from AffordPose JSON format"""
    vertices = []
    faces = []
    
    if isinstance(obj_mesh_data, str):
        lines = obj_mesh_data.split('\n')
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
    elif isinstance(obj_mesh_data, list):
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
    
    # If no faces were found, generate simple triangulation
    if len(faces) == 0 and len(vertices) > 2:
        for i in range(0, len(vertices) - 2, 3):
            faces.append([i, i+1, i+2])
    
    return vertices, faces


def extract_unique_objects(args):
    """Extract unique object meshes from AffordPose JSON files"""
    affordpose_dir = Path(args.affordpose_data_dir)
    export_dir = Path(args.output_dir) / "original"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique objects
    unique_objects = {}
    
    print("Scanning AffordPose dataset for unique objects...")
    for category_dir in sorted(affordpose_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.endswith('.7z'):
            continue
        
        category = category_dir.name
        print(f"Processing category: {category}")
        
        for obj_instance_dir in sorted(category_dir.iterdir()):
            if not obj_instance_dir.is_dir():
                continue
            
            obj_instance = obj_instance_dir.name
            
            for action_dir in sorted(obj_instance_dir.iterdir()):
                if not action_dir.is_dir():
                    continue
                
                action = action_dir.name
                
                # Process first JSON file to get object mesh
                json_files = sorted(action_dir.glob('*.json'))
                if len(json_files) == 0:
                    continue
                
                # Use first JSON file as representative
                json_path = json_files[0]
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                obj_mesh_data = data.get('object_mesh', [])
                vertices, faces = parse_obj_mesh(obj_mesh_data)
                
                # Create unique object ID
                obj_id = f"{category}_{obj_instance}"
                
                if obj_id not in unique_objects:
                    unique_objects[obj_id] = {
                        'category': category,
                        'instance': obj_instance,
                        'vertices': vertices,
                        'faces': faces
                    }
    
    print(f"Found {len(unique_objects)} unique objects")
    
    # Export unique objects - skip if already exists
    print("Exporting unique objects...")
    skipped_count = 0
    for obj_id, obj_data in tqdm(unique_objects.items(), desc="Exporting objects"):
        obj_path = export_dir / f"{obj_id}.obj"
        if obj_path.exists():
            skipped_count += 1
            continue
        try:
            # Create mesh
            mesh = trimesh.Trimesh(vertices=obj_data['vertices'], faces=obj_data['faces'])
            # Export to OBJ
            mesh.export(str(obj_path), include_normals=False)
        except Exception as e:
            print(f"Error exporting {obj_id}: {e}")
    
    print(f"Exported {len(unique_objects)-skipped_count} objects (skipped {skipped_count} to {export_dir}")
    return len(unique_objects)


def watertight_one(oid, omesh, export_dir):
    """Make one mesh watertight using ManifoldPlus"""
    out_path = os.path.join(export_dir, f"{oid}.obj")
    if os.path.exists(out_path):
        return
    
    # export and skip materials
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
        # remove temp file
        os.remove(temp_path)
    except Exception as e:
        # dump the error message to a file
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
    print(f"Checking if {res_binvox_path} exists...")
    if not os.path.exists(res_binvox_path):
        print(f"File {res_binvox_path} does not exist!")
    else:
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
    parser = argparse.ArgumentParser(description="Process AffordPose objects for physics evaluation (with skip support)")
    parser.add_argument("--affordpose_data_dir", type=str, 
                       default="/home/zhenhua/grasp/AffordPose-main/AffordPose Data/",
                       help='Path to AffordPose Data directory')
    parser.add_argument("--output_dir", type=str, 
                       default="/home/zhenhua/grasp/AffordPose-main/postprocess_object/",
                       help='Output directory for processed objects')
    parser.add_argument("--stage", type=str, default="all", 
                       choices=["extract", "watertight", "voxel", "vhacd", "all"],
                       help='Processing stage')
    parser.add_argument("--n_jobs", type=int, default=8, help='Number of parallel jobs')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "extract" or args.stage == "all":
        print("=" * 60)
        print("Stage 1: Extracting unique objects from AffordPose dataset")
        print("=" * 60)
        extract_unique_objects(args)
    
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
