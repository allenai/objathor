import argparse
import glob
import os
import shutil
import stat
import subprocess
import sys
from sys import platform
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh

try:
    from objathor.asset_conversion.util import (
        get_existing_thor_asset_file_path,
        load_existing_thor_asset_file,
        save_thor_asset_file,
    )
except ImportError:
    try:
        from util import (
            get_existing_thor_asset_file_path,
            load_existing_thor_asset_file,
            save_thor_asset_file,
        )
    except ImportError as e:
        sys.exit(f"{e} Error impoerint package utils.")

HOW_MANY_MESSED_UP_MESH = 0

_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
VHACD_PATH = None

if platform == "linux" or platform == "linux2":
    VHACD_PATH = os.path.join(_FILE_DIR, "linux", "TestVHACD")
    # not sure why/when this worked, command requires specific flags
    # command = [os.path.join(file_dir, "linux", "TestVHACD"), obj_file, "-h", str(num_colliders)]
elif platform == "darwin":
    VHACD_PATH = os.path.join(_FILE_DIR, "osx", "TestVHACD")
elif platform.startswith("win"):
    VHACD_PATH = os.path.join(_FILE_DIR, "win", "TestVHACD.exe")
else:
    raise NotImplementedError


def vhacd_init(vhacd_path):
    if not os.path.exists(vhacd_path):
        download_vhacd(os.path.abspath(os.path.join(os.path.dirname(vhacd_path), "..")))
    if not platform.startswith("win"):
        try:
            st = os.stat(vhacd_path)
            os.chmod(vhacd_path, st.st_mode | stat.S_IEXEC)
        except OSError as e:
            if e.errno == 30:
                # READ ONLY FILE SYSTEM
                # Copy file to cache directory
                print("Copying VHACD to cache directory")
                _cache_dir = os.path.join(os.environ["HOME"], ".vhacd")
                os.makedirs(_cache_dir, exist_ok=True)
                _new_path = os.path.join(_cache_dir, "TestVHACD")
                shutil.copyfile(vhacd_path, _new_path)
                st = os.stat(_new_path)
                os.chmod(_new_path, st.st_mode | stat.S_IEXEC)
                vhacd_path = _new_path
            else:
                raise


def download_vhacd(out_path):
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile

    url = "https://objathor-vhacd-bin.s3.us-west-2.amazonaws.com/"

    if platform == "linux" or platform == "linux2":
        url = f"{url}VHACD_linux.zip"
    elif platform == "darwin":
        # TODO distinguish intel vs M2
        url = f"{url}VHACD_osx.zip"
    elif platform.startswith("win"):
        url = f"{url}VHACD_win.zip"
    else:
        raise NotImplementedError

    try:
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(os.path.abspath(out_path))
    except Exception as e:
        print(
            f"Error downloading and unziping VHACD. {e}. If you can't reach `{url}` make sure to download the VHACD binary and place in `{VHACD_PATH}`."
        )


def decompose_obj(
    file_path: Optional[str],
    decomposed_file_paths: Optional[List[str]] = None,
    output_file_prefix="decomp_",
):
    if file_path is None:
        decomposed_files = []
        file_path = decomposed_file_paths[0]
        for file_path in decomposed_file_paths:
            with open(file_path, "r") as f:
                decomposed_files.append(f.read())
    else:
        assert decomposed_file_paths is None

        with open(file_path, "r") as f:
            lines = [l for l in f]

        decomposed_files = []

        current_file = []
        for l in lines:
            if l[:2] == "o ":
                if len(current_file) > 0:
                    decomposed_files.append(current_file)
                current_file = []

            current_file.append(l)

        if len(current_file) > 0:
            decomposed_files.append(current_file)

    vertices_so_far = 0
    for i in range(len(decomposed_files)):
        current_file = decomposed_files[i]
        file_to_write = os.path.join(
            os.path.dirname(file_path), f"{output_file_prefix}{i}.obj"
        )
        with open(file_to_write, "w") as f:
            for l in current_file:
                if l[:2] == "f ":
                    line_to_write = l.replace("\n", "").split(" ")[1:]
                    # print(f"line: {line_to_write}")
                    vertex_numbers = [
                        int(x) - vertices_so_far for x in line_to_write if x != ""
                    ]
                    assert len(vertex_numbers) == 3
                    l = f"f {vertex_numbers[0]} {vertex_numbers[1]} {vertex_numbers[2]}\n"
                f.write(l)

        current_vertices = len([l for l in current_file if l[:2] == "v "])
        vertices_so_far += current_vertices


def get_colliders(
    obj_file: str, num_colliders: int, capture_out: bool, **kwargs
) -> List[Dict[str, Any]]:
    if not capture_out:
        print("processing... ", obj_file)

    result_info = {}

    # command for old binary
    output_obj_name = "decomp.obj"
    extra_args = [f"--{k} {str(v)}" for k, v in kwargs.items()]

    if not capture_out:
        print(f"Extra args: {extra_args}")
    # "--resolution", str(6e6),
    command = [
        VHACD_PATH,
        "--maxhulls",
        str(num_colliders),
        "--input",
        obj_file,
        "--output",
        output_obj_name,
        *extra_args,
    ]

    if capture_out:
        VHACD_result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        result_info["stderr"] = VHACD_result.stderr
        result_info["stdout"] = VHACD_result.stdout
    else:
        VHACD_result = subprocess.run(command)
    result_info["VHACD_returncode"] = VHACD_result.returncode

    if not os.path.exists(output_obj_name):
        result_info["failed"] = True
        result_info[
            "stderr"
        ] = f"VHACD did not generate 'decomp.obj'. Unsuccessfull run of command: {command}"
        return [], result_info
    decompose_obj(output_obj_name)

    # try:

    # TODO: refine error checking
    os.remove(output_obj_name)
    if os.path.exists("decomp.stl"):
        os.remove("decomp.stl")
    # except Exception:
    #     print('NO DECOMP FILE WAS GENERATED', obj_file)
    #     result_info["failed"] = True
    #     return [], result_info
    decomps = glob.glob("decomp_*.obj")
    colliders = []
    # print(decomps, num_colliders, "I HATE THIS")
    for decomp in decomps:
        colliders.append(trimesh.load(decomp))
        os.remove(decomp)
    out = []
    # print(f"Colliders {len(colliders)}")
    for collider in colliders:
        try:
            collider.vertices
        except Exception:
            print("Collider failed", collider)
            global HOW_MANY_MESSED_UP_MESH
            HOW_MANY_MESSED_UP_MESH += 1
            print("HOW_MANY_MESSED_UP_MESH", HOW_MANY_MESSED_UP_MESH)
            # result_info["failed"] = True
            # result_info["HOW_MANY_MESSED_UP_MESH"] = HOW_MANY_MESSED_UP_MESH
            continue
        out.append(
            {
                "vertices": [
                    dict(x=x, y=y, z=z) for x, y, z in collider.vertices.tolist()
                ],
                "triangles": np.array(collider.faces).reshape(-1).tolist(),
            }
        )
    return out, result_info


def get_colliders_vhacd41(
    obj_file: str, num_colliders: int, capture_out: bool, **kwargs
) -> List[Dict[str, Any]]:
    # Using version 4.1 VHACD binary from https://github.com/kmammou/v-hacd
    assert num_colliders < 1000

    if not capture_out:
        print("processing... ", obj_file)

    result_info = {}

    extra_args = [f"-{k} {str(v)}" for k, v in kwargs.items()]

    if not capture_out:
        print(f"Extra args: {extra_args}")

    command = [
        VHACD_PATH,
        obj_file,
        "-h",
        str(num_colliders),
        "-o",
        "obj",
        *extra_args,
    ]

    if capture_out:
        VHACD_result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        result_info["stderr"] = VHACD_result.stderr
        result_info["stdout"] = VHACD_result.stdout
    else:
        VHACD_result = subprocess.run(command)

    result_info["VHACD_returncode"] = VHACD_result.returncode

    should_exist = obj_file.replace(".obj", "000.obj")
    if not os.path.exists(should_exist):
        result_info["failed"] = True
        result_info[
            "stderr"
        ] = f"VHACD did not generate '{should_exist}'. Unsuccessful run of command: {command}"
        return [], result_info

    vhacd_outputs = glob.glob(obj_file.replace(".obj", "???.obj"))

    colliders = []
    for decomp in vhacd_outputs:
        colliders.append(trimesh.load(decomp))
        os.remove(decomp)

    out = []
    # print(f"Colliders {len(colliders)}")
    for collider in colliders:
        try:
            collider.vertices
        except Exception:
            print("Collider failed", collider)
            global HOW_MANY_MESSED_UP_MESH
            HOW_MANY_MESSED_UP_MESH += 1
            print("HOW_MANY_MESSED_UP_MESH", HOW_MANY_MESSED_UP_MESH)
            # result_info["failed"] = True
            # result_info["HOW_MANY_MESSED_UP_MESH"] = HOW_MANY_MESSED_UP_MESH
            continue
        out.append(
            {
                "vertices": [
                    dict(x=x, y=y, z=z) for x, y, z in collider.vertices.tolist()
                ],
                "triangles": np.array(collider.faces).reshape(-1).tolist(),
            }
        )
    return out, result_info


def set_colliders(
    obj_file: str, num_colliders: int = 4, capture_out=False, **kwargs
) -> Dict[str, Any]:
    obj_file_dir = os.path.dirname(obj_file)
    uid = os.path.splitext(os.path.basename(obj_file))[0]

    annotations_file = get_existing_thor_asset_file_path(
        out_dir=obj_file_dir, asset_id=uid
    )
    if not capture_out:
        print(f"--- setting colliders... {annotations_file}")

    annotations = load_existing_thor_asset_file(out_dir=obj_file_dir, object_name=uid)

    if "colliders" in annotations:
        msg = f"colliders already exist for {obj_file}"
        print(msg)
        return {"failed": True, "stdout": msg}
    colliders, result_info = get_colliders(
        obj_file, num_colliders, capture_out=capture_out, **kwargs
    )

    annotations["colliders"] = colliders

    save_thor_asset_file(asset_json=annotations, save_path=annotations_file)
    return result_info


def generate_colliders(
    source_directory, num_colliders=4, delete_objs=False, capture_out=False, **kwargs
):
    vhacd_init(VHACD_PATH)

    obj_files = glob.glob(os.path.join(source_directory, "*.obj"))
    # print(obj_files)

    info = {}

    for obj_file in obj_files:
        uid = os.path.splitext(os.path.basename(obj_file))[0]
        result_info = set_colliders(
            obj_file=obj_file,
            num_colliders=num_colliders,
            capture_out=capture_out,
            **kwargs,
        )
        info[uid] = result_info

    # delete the obj files and the texture files
    if delete_objs:
        for file in obj_files:  # + texture_files:
            os.remove(file)
    return info


if __name__ == "__main__":
    BASE_PATH = os.getcwd()
    print("---- Running generate_colliders")
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=f"{BASE_PATH}/glbs/post")
    parser.add_argument("--num_colliders", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    generate_colliders(args.folder, args.num_colliders, args.clean)
