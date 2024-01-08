import os
import sys
import json
import platform

try:
    import bpy
except ImportError:
    raise
import subprocess

# copy object_consolidater.py to where blender project file is
project_dir = os.path.dirname(bpy.data.filepath)
print(f"Project dir: {project_dir}")
root_proj_dir = "C:/Users/Eli/Documents/GitHub/objathor/asset_conversion"
if not project_dir in sys.path:
    sys.path.append(project_dir)

print(f"Root proj dir: {root_proj_dir}")
if not root_proj_dir in sys.path:
    sys.path.append(root_proj_dir)

if not project_dir in sys.path:
    sys.path.append(project_dir)

python_modules_path = "C:/Users/Eli/AppData/Roaming/Python/Python310/site-packages"
sys.path.append(python_modules_path)

print(sys.prefix)
print(os.name)
python_filename = "python"
if platform.system() == "Darwin":
    python_filename = "python3.9"
elif platform.system() == "Windows":
    python_filename = "python.exe"

python_exe = os.path.join(sys.prefix, "bin", python_filename)


## upgrade pip
# subprocess.call([python_exe, "-m", "ensurepip"])
# subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
#
## install required packages
# subprocess.call([python_exe, "-m", "pip", "install", "trimesh"])
# subprocess.call([python_exe, "-m", "pip", "install", "numpy"])
# subprocess.call([python_exe, "-m", "pip", "install", "objaverse"])

import trimesh
import numpy as np

import object_consolidater
import util
import imp

imp.reload(object_consolidater)
imp.reload(util)


def get_colliders(collider):
    return [
        {
            "vertices": [dict(x=x, y=y, z=z) for x, y, z in collider.vertices.tolist()],
            "triangles": np.array(collider.faces).reshape(-1).tolist(),
        }
    ]


if __name__ == "__main__":
    print("dir: " + os.getcwd())

    obj_name = "ffeabdae926d4cf798cd82da55ebd222"

    objaverse_root = "C:/Users/Eli/Desktop/Maya_Projects/2022-10-14 - Blender Optimizations/glbs/4_new_glbs/fire_hydrant"
    thor_unity_path = "C:/Users/Eli/Documents/GitHub/ai2thor_6/unity"

    # objaverse_root = "/Users/alvaroh/.objaverse/hf-objaverse-v1/glbs/000-067"
    # thor_unity_path = "/Users/alvaroh/ai2/ai2thor/unity"

    # WE'RE TEMPORARILY DISREGARDING THIS ANNOTATIONS_FILE
    annotations_file = ""
    object_path = os.path.join(objaverse_root, f"{obj_name}.glb")
    output_dir = os.path.join(thor_unity_path, "debug", obj_name)
    house_output_file = os.path.join(
        thor_unity_path, "Assets/Resources/rooms", f"{obj_name}.json"
    )
    engine = "CYCLES"
    save_obj = True

    if not os.path.exists(object_path):
        import objaverse

        objects = objaverse.load_objects(uids=[obj_name], download_processes=1)
        object_path = objects[obj_name]

    object_consolidater.glb_to_thor(
        object_path=object_path,
        output_dir=output_dir,
        engine=engine,
        annotations_file=annotations_file,
        save_obj=save_obj,
        save_as_json=True,
        relative_texture_paths=False,
    )

    instance_id = "asset_0"
    skybox_color = (1, 1, 1)

    output_json = os.path.join(output_dir, f"{obj_name}.json")

    # out_obj = os.path.join(output_dir, f"{obj_name}.obj")
    # collider = trimesh.load(out_obj)

    import colliders.generate_colliders

    imp.reload(colliders.generate_colliders)
    print("Generating colliders with library....")
    extra_args = dict(
        # resolution=1000000
    )
    colliders.generate_colliders.generate_colliders(
        output_dir, num_colliders=15, **extra_args
    )

    util.add_visualize_thor_actions(
        asset_id=obj_name,
        instance_id=obj_name,
        asset_dir=output_dir,
        instance_id=instance_id,
        house_path="data/empty_house.json",
        house_skybox_color=skybox_color,
    )
