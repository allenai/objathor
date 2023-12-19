import os
import sys
import json
import platform
import logging

try:
    import bpy
except ImportError:
    raise
import subprocess

# pip install numpy
# pip install objaverse_to_thor
# pip install trimesh

import trimesh
import numpy as np

import object_consolidater
import util

logger = logging.getLogger()

logger.setLevel(logging.INFO)
print("----- " + __name__)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.ERROR)
logger.addHandler(handler)

project_dir = os.path.dirname(bpy.data.filepath)
print(f"Project dir: {project_dir}")

dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"File dir: {dir_path}")

if not dir_path in sys.path:
    print("adding file dir to path")
    sys.path.append(dir_path)


# if not project_dir in sys.path:
#     print("adding to path")
#     sys.path.append(project_dir)
def get_colliders(collider):
    return [
        {
            "vertices": [dict(x=x, y=y, z=z) for x, y, z in collider.vertices.tolist()],
            "triangles": np.array(collider.faces).reshape(-1).tolist(),
        }
    ]


if __name__ == "__main__":
    print("dir: " + os.getcwd())
    object_ids = sys.argv[1].split(",")

    # objaverse_root = "C:/Users/Eli/Desktop/Maya_Projects/2022-10-14 - Blender Optimizations/sofa_ring_glb/glbs/"
    # thor_unity_path = "C:/Users/Eli/Documents/GitHub/ai2thor_3/unity"

    # objaverse_root = "/Users/alvaroh/.objaverse_to_thor/hf-objaverse_to_thor-v1/glbs/000-067"
    # thor_unity_path = "/Users/alvaroh/ai2/ai2thor/unity"

    objaverse_root = sys.argv[2]
    print(
        f"Running pipeline for object '{object_ids}' and writing to  '{objaverse_root}'"
    )

    for object_id in object_ids:
        # annotations_file = "annotations/objaverse_thor_v0p95.json"
        annotations_file = ""
        object_path = os.path.join(objaverse_root, f"{object_id}.glb")
        output_dir = os.path.join(objaverse_root, object_id)
        # house_output_file = os.path.join(thor_unity_path, "Assets/Resources/rooms", f"{obj_name}.json")
        engine = "CYCLES"
        save_obj = True

        if not os.path.exists(object_path):
            import objaverse

            objects = objaverse.load_objects(uids=[object_id])
            object_path = objects[object_id]

            print(f"Donwloaded object at: {object_path}")

        object_consolidater.glb_to_thor(
            object_path=object_path,
            output_dir=output_dir,
            engine=engine,
            annotations=annotations_file,
            save_obj=save_obj,
            save_as_json=True,
        )

        # from util import load_existing_thor_obj_file
        # obj_file = load_existing_thor_obj_file(output_dir, object_id)
        # obj_file["TexturePath"]

        # For compressing textures
        # from util import compress_image_to_ssim_threshold, load_existing_thor_obj_file, save_thor_obj_file
        # obj_file = load_existing_thor_obj_file(output_dir, object_id)
        # for k in ["albedo", "emission"]:
        #     compress_image_to_ssim_threshold(
        #         input_path=os.path.join(output_dir, f"{k}.png"),
        #         output_path=os.path.join(output_dir, f"{k}.jpg"),
        #         threshold=0.95,
        #     )
        #     os.remove(os.path.join(output_dir, f"{k}.png"))
        #     obj_file[f"{k}TexturePath"] = obj_file[f"{k}TexturePath"].replace(".png", ".jpg")
        # save_thor_obj_file(obj_file, object_path)

        print(f"path : {os. getcwd()}")
        print(sys.path)
        import colliders.generate_colliders as coll

        # import colliders.generate_colliders
        # imp.reload(colliders.generate_colliders)
        # print("Generating colliders with library....")
        extra_args = dict(resolution=1000000)
        coll.generate_colliders(output_dir, num_colliders=15, **extra_args)

        # util.add_visualize_thor_actions(
        #     asset_id=obj_name,
        #     asset_dir=output_dir,
        #     instance_id=instance_id,
        #     house_path="test_houses/empty_house.json",
        #     house_skybox_color=skybox_color
        # )
