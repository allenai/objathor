import argparse


def convert_obj_to_glb(obj_path: str, glb_path: str):
    import bpy

    # Clear existing data
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the OBJ file
    bpy.ops.import_scene.obj(filepath=obj_path)

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
