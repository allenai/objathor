import argparse
import os


def convert_obj_to_glb(obj_path: str, glb_path: str):

    try:
        import bpy
    except ImportError as e:
        raise ImportError(
            f"{e}: Blender is not installed, make sure to either run 'pip install bpy' to install it as a module or as an application https://docs.blender.org/manual/en/latest/getting_started/installing/index.html"
        )

    bpy.ops.object.delete(use_global=False)

    bpy.ops.wm.obj_import(
        filepath=obj_path,
        directory=os.path.dirname(obj_path),
        files=[{"name": os.path.basename(obj_path)}],
    )

    # Export to GLB
    bpy.ops.export_scene.gltf(filepath=glb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_path",
        type=str,
        required=True,
        help="Path to the .obj file",
    )
    parser.add_argument(
        "--glb_path",
        type=str,
        required=True,
        help="Path to save the glb",
    )
    args = parser.parse_args()

    convert_obj_to_glb(
        obj_path=args.obj_path,
        glb_path=args.glb_path,
    )
