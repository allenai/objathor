import argparse
import json
import logging
import multiprocessing
import os
import random
import subprocess
import sys
import time
import traceback
from typing import Any, List, Dict

import numpy as np
import objaverse

from objathor.asset_conversion.colliders.generate_colliders import generate_colliders
from objathor.constants import ABS_PATH_OF_OBJATHOR, THOR_COMMIT_ID

# shared library
from objathor.asset_conversion.util import (
    add_visualize_thor_actions,
    get_blender_installation_path,
    create_asset,
    view_asset_in_thor,
    OrderedDictWithDefault,
    get_existing_thor_asset_file_path,
    compress_image_to_ssim_threshold,
    load_existing_thor_asset_file,
    save_thor_asset_file,
    get_extension_save_path,
    add_default_annotations,
)

FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger(__name__)


def glb_to_thor(
    glb_path: str,
    annotations_path: str,
    object_out_dir: str,
    uid: str,
    failed_objects: OrderedDictWithDefault,
    capture_stdout=False,
    timeout=None,
    generate_obj=True,
    save_as_json=False,
    relative_texture_paths=True,
    run_blender_as_module=False,
    blender_instalation_path=None,
):
    os.makedirs(object_out_dir, exist_ok=True)

    if not run_blender_as_module:
        command = (
            f"{blender_instalation_path if blender_instalation_path is not None else get_blender_installation_path()}"
            f" --background"
            f" --python {os.path.join(ABS_PATH_OF_OBJATHOR, 'asset_conversion', 'object_consolidater.py')}"
            f" --"
            f' --object_path="{os.path.abspath(glb_path)}"'
            f' --output_dir="{os.path.abspath(object_out_dir)}"'
            f' --annotations="{annotations_path}"'
        )
    else:
        command = (
            f"python"
            f" -m"
            f" objathor.asset_conversion.object_consolidater"
            f" --"
            f' --object_path="{os.path.abspath(glb_path)}"'
            f' --output_dir="{os.path.abspath(object_out_dir)}"'
            f' --annotations="{annotations_path}"'
        )

    if generate_obj:
        command = command + " --obj"

    if save_as_json:
        command = command + " --save_as_json"

    if relative_texture_paths:
        command = command + " --relative_texture_paths"

    if not capture_stdout:
        print(f"For {uid}, running command: {command}")

    process = None
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        out, _ = process.communicate(timeout=timeout)
        out = out.decode()
        result_code = process.returncode
        if result_code != 0:
            raise subprocess.CalledProcessError(result_code, command)
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
            process.wait(timeout=timeout)
        result_code = -1
        out = f"Command timed out, command: {command}"
    except subprocess.CalledProcessError as e:
        result_code = e.returncode
        print(f"Blender call error: {e.output}, {out}")
        out = f"{out}, Exception: {e.output}"
    except Exception as e:
        result_code = e.returncode
        print(f"Blender process error: {e.output}")
        out = e.output

    if not capture_stdout:
        print(f"Exited with code {result_code}")

    success = result_code == 0

    if success:
        if not capture_stdout:
            print(f"---- Command ran successfully for {uid} at path {glb_path}")

    else:
        failed_objects[uid]["blender_process_return_fail"] = True
        failed_objects[uid]["blender_output"] = out if out else ""

    try:
        # The below compresses textures using the structural similarity metric
        # This is a lossy compression, but attempts to preserve the visual quality
        thor_obj_path = get_existing_thor_asset_file_path(object_out_dir, uid)

        # TODO: here optimize to remove needing to decompress and change references,
        # always export as json from blender pipeline and change to desired compression here
        asset_json = load_existing_thor_asset_file(os.path.abspath(object_out_dir), uid)

        save_dir = os.path.dirname(thor_obj_path)
        for k in ["albedo", "normal", "emission"]:
            compress_image_to_ssim_threshold(
                input_path=os.path.join(save_dir, f"{k}.png"),
                output_path=os.path.join(save_dir, f"{k}.jpg"),
                threshold=0.95,
            )
            os.remove(os.path.join(save_dir, f"{k}.png"))
            asset_json[f"{k}TexturePath"] = asset_json[f"{k}TexturePath"].replace(
                ".png", ".jpg"
            )

        y_rot = compute_thor_rotation_to_obtain_min_bounding_box(
            asset_json["vertices"], max_deg_change=45, increments=91
        )
        asset_json["yRotOffset"] = y_rot
        print(f"Pose adjusted by {y_rot:.2f} degrees ({uid})")

        save_thor_asset_file(asset_json, thor_obj_path)

    except Exception as e:
        logger.error(f"Exception: {e}")
        failed_objects[uid]["blender_process_crash"] = False
        failed_objects[uid]["image_compress_fail"] = True
        #  Do we want this? confuses failure reason
        # failed_objects[uid]["blender_output"] = out
        failed_objects[uid]["exception"] = f"{e}"
        success = False
    return success


def compute_axis_aligned_bbox_volume(
    vertices: np.ndarray,
):
    assert vertices.shape[0] == 3

    mins = np.min(vertices, axis=1)
    maxes = np.max(vertices, axis=1)
    return float((maxes - mins).prod())


def compute_thor_rotation_to_obtain_min_bounding_box(
    vertices: List[Dict[str, float]],
    max_deg_change: float,
    increments: int = 31,
    bias_for_no_rotation: float = 0.01,
):
    """
    Computes the (approximate) rotation in [-max_deg_change, max_deg_change] around the y-axis that
     minimizes the volume of the axis aligned bounding box.

    :param vertices: list of vertices
    :param max_deg_change: maximum rotation in degrees
    :param increments: number of increments to try
    :param bias_for_no_rotation: Bias to use no rotation by rescaling no rotation volume by (1-bias_for_no_rotation)
    """
    assert max_deg_change >= 0
    assert (
        increments >= 1 and increments % 2 == 1
    ), "Increments must be non-negative and odd"

    vertices_arr = np.array(
        [[v["x"], v["y"], v["z"]] for v in vertices], dtype=np.float32
    )
    vertices_arr = vertices_arr.transpose((1, 0))

    def get_rotmat(t):
        return np.array(
            [  # rotation matrix for rotating around the y axis
                [np.cos(t), 0, -np.sin(t)],
                [0, 1, 0],
                [np.sin(t), 0, np.cos(t)],
            ]
        )

    max_rad_change = np.deg2rad(max_deg_change)
    thetas = np.linspace(start=-max_rad_change, stop=max_rad_change, num=increments)
    volumes = []
    for theta in thetas:
        volumes.append(
            compute_axis_aligned_bbox_volume(np.matmul(get_rotmat(theta), vertices_arr))
        )

    volumes[len(volumes) // 2] *= 1 - bias_for_no_rotation

    min_volume_theta = thetas[np.argmin(volumes)]

    # NEED TO NEGATE TO GET THE RIGHT ROTATION AS UNITY TREATS CLOCKWISE
    # ROTATIONS AS POSITIVE
    return -np.rad2deg(min_volume_theta)


def obj_to_colliders(
    uid: str,
    object_out_dir: str,
    max_colliders: int,
    capture_stdout: bool,
    failed_objects: OrderedDictWithDefault,
    delete_objs=False,
    **kwargs,
):
    try:
        result_info = generate_colliders(
            source_directory=object_out_dir,
            num_colliders=max_colliders,
            delete_objs=delete_objs,
            capture_out=capture_stdout,
            **kwargs,
        )
        if "failed" in result_info[uid] and result_info[uid]["failed"]:
            failed_objects[uid]["failded_generate_colliders"] = True
            failed_objects[uid]["generate_colliders_info"] = result_info[uid]
            return False
        else:
            return True

    except Exception as e:
        failed_objects[uid]["failded_generate_colliders"] = True
        print("Exception while running 'generate_colliders'")
        if hasattr(e, "message"):
            print(e.message)
        else:
            print(e)
        failed_objects[uid]["generate_colliders_exception"] = traceback.format_exc()
        print(traceback.format_exc())
        return False


def validate_in_thor(
    controller: Any,
    asset_dir: str,
    asset_name: str,
    output_dir: str,
    failed_objects: OrderedDictWithDefault,
    skip_images=False,
    skybox_color=(255, 255, 255),
    load_file_in_unity=False,
    extension=None,
):
    evt = None
    try:
        evt = create_asset(
            thor_controller=controller,
            asset_directory=asset_dir,
            asset_id=asset_name,
            load_file_in_unity=load_file_in_unity,
            extension=extension,
        )
        if not evt.metadata["lastActionSuccess"]:
            failed_objects[asset_name] = {
                "failed_create_asset_in_thor": True,
                "lastAction": controller.last_action,
                "errorMessage": evt.metadata["errorMessage"],
            }
            return False, None

        asset_metadata = evt.metadata["actionReturn"]
        angle_increment = 45
        angles = [n * angle_increment for n in range(0, round(360 / angle_increment))]
        axes = [(0, 1, 0), (1, 0, 0)]
        rotations = [(x, y, z, degrees) for degrees in angles for (x, y, z) in axes]

        if not skip_images:
            # print(rotations)
            evt = view_asset_in_thor(
                asset_name,
                controller,
                output_dir,
                rotations=rotations,
                skybox_color=skybox_color,
            )
            if not evt.metadata["lastActionSuccess"]:
                failed_objects[asset_name] = {
                    "failed_thor_view_asset_in_thor": True,
                    "lastAction": controller.last_action,
                    "errorMessage": evt.metadata["errorMessage"],
                }
                return False, asset_metadata
        return True, asset_metadata
    except Exception as e:
        print(f"Exception: {e}")
        print(traceback.format_exc())
        failed_objects[asset_name] = {
            "failed_thor_validate_in_thor": True,
            "stderr": traceback.format_exc(),
            "lastAction": controller.last_action,
            "errorMessage": evt.metadata["errorMessage"] if evt else "",
        }
        return False, None


# TODO entrypoint for package version of blender, document and type hints, same as old cli API
# needs to handle stop condition
def run_pipeline(
    output_dir,
    object_ids,
    annotations="",
    live=False,
    max_colliders=4,
    skip_glb=False,
    delete_objs=False,
    skip_colliders=False,
    skip_thor_creation=False,
    skip_thor_visualization=False,
    add_visualize_thor_actions=False,
    verbose=False,
    width=300,
    height=300,
    skybox_color=(0, 0, 0),
    save_as_json=True,
    relative_texture_paths=True,
    **kwargs,
):
    if verbose:
        # TODO use logger instead of print
        logger.setLevel(logging.DEBUG)

    # glb_to_thor(
    #             object_path=glb_path,
    #             output_dir=asset_out_dir,
    #             annotations=annotations_path,
    #             save_obj=True,
    #             save_as_json=True,
    #             relative_texture_paths=True
    #         )
    pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    print("--------- argv")
    print(argv)

    parser.add_argument("--output_dir", type=str, default="./output", required=True)
    parser.add_argument(
        "--object_ids",
        type=str,
        default="",
        help="Comma separated list of object ids to process, overrides number.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="",
    )
    parser.add_argument(
        "--number", type=int, default=1, help="Number of random objects to take."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Runs subcommands with live output, does not store outputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--max_colliders",
        type=int,
        default=4,
        help="Maximum hull colliders for collider extraction with TestVHACD.",
    )
    parser.add_argument(
        "--skip_glb", action="store_true", help="Skips glb to json generation."
    )
    parser.add_argument(
        "--delete_objs",
        action="store_true",
        help="Deletes objs after generating colliders.",
    )
    parser.add_argument(
        "--skip_colliders",
        action="store_true",
        help="Skips obj to json collider generation.",
    )
    parser.add_argument(
        "--skip_thor_creation",
        action="store_true",
        help="Skips THOR asset creation and visualization.",
    )
    parser.add_argument(
        "--skip_thor_visualization",
        action="store_true",
        help="Skips THOR asset visualization.",
    )

    parser.add_argument(
        "--add_visualize_thor_actions",
        action="store_true",
        help="Adds house creation with single object and look at object center actions to json.",
    )

    parser.add_argument(
        "--width", type=int, default=300, help="Width of THOR asset visualization."
    )
    parser.add_argument(
        "--height", type=int, default=300, help="Height of THOR asset visualization."
    )
    parser.add_argument(
        "--skybox_color",
        type=str,
        default="255,255,255",
        help="Comma separated list off r,g,b values for skybox thor images.",
    )

    parser.add_argument(
        "--save_as_pkl", action="store_true", help="Saves asset as pickle gz."
    )

    parser.add_argument(
        "--absolute_texture_paths",
        action="store_true",
        help="Saves textures as absolute paths.",
    )

    parser.add_argument(
        "--extension",
        choices=[".json", ".pkl.gz", ".msgpack", ".msgpack.gz", ".gz"],
        default=".json",
    )

    parser.add_argument(
        "--send_asset_to_controller",
        action="store_true",
        help="Sends all the asset data to the thor controller.",
    )

    parser.add_argument(
        "--blender_as_module",
        action="store_true",
        help="Runs blender as a module. Requires bpy to be installed in python environment.",
    )

    parser.add_argument(
        "--keep_json_asset",
        action="store_true",
        help="Wether it keeps the intermediate .json asset file when storing in a different format to json.",
    )

    found_blender = False
    try:
        get_blender_installation_path()
        found_blender = True
    except:
        pass

    parser.add_argument(
        "--blender_installation_path",
        type=str,
        default=None,
        help="Blender installation path, when blender_as_module = False and we cannot find the installation path automatically.",
        required=(not found_blender) and "--blender_as_module" not in argv,
    )

    parser.add_argument(
        "--thor_platform",
        type=str,
        default="OSXIntel64" if sys.platform == "darwin" else "CloudRendering",
        help="THOR platform to use.",
        choices=["CloudRendering", "OSXIntel64"],  # Linux64
    )

    # Necessary for mesh decomposition to generate colliders
    # parser.add_argument(
    #     "--obj", action="store_true", help="Saves obj version of asset."
    # )

    args, unknown = parser.parse_known_args(argv)
    extra_args_keys = []
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            print(arg)
            extra_args_keys.append(arg.split("=")[0].removeprefix("--"))
            parser.add_argument(arg.split("=")[0], type=str)

    args = parser.parse_args(argv)
    print(args)

    annotations_path = args.annotations
    object_number = args.number
    max_colliders = args.max_colliders
    out_metadata_filename = "thor_metadata.json"

    output_dir = args.output_dir
    report_out_path = os.path.join(output_dir, "failed_objects.json")

    if args.verbose:
        # TODO use logger instead of print
        logger.setLevel(logging.DEBUG)

    failed_objects = OrderedDictWithDefault(dict)

    process_count = multiprocessing.cpu_count()

    if args.object_ids != "":
        selected_uids = args.object_ids.split(",")
    else:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        selected_uids = sorted(random.sample(annotations.keys(), object_number))

    print(f"--- Selected uids: {selected_uids}")

    objects = objaverse.load_objects(
        uids=selected_uids, download_processes=process_count
    )

    controller = None
    start_process_time = time.perf_counter()

    for uid, glb_path in objects.items():
        start_obj_time = time.perf_counter()
        asset_out_dir = os.path.join(output_dir, uid)

        success = True
        if not args.skip_glb:
            print("GLB to THOR starting...")
            start = time.perf_counter()
            success = glb_to_thor(
                glb_path=glb_path,
                annotations_path=args.annotations,
                object_out_dir=asset_out_dir,
                uid=uid,
                failed_objects=failed_objects,
                capture_stdout=not args.live,
                generate_obj=True,
                save_as_json=not args.save_as_pkl,
                relative_texture_paths=not args.absolute_texture_paths,
                run_blender_as_module=args.blender_as_module,
                blender_instalation_path=args.blender_installation_path,
            )
            end = time.perf_counter()
            print(f"GLB to THOR success: {success}. Runtime: {end-start}s")

            # print(f"uid in failed {uid in failed_objects} 'Progress: 100.00%' in failed_objects[uid]['blender_output'] {'Progress: 100.00%' in failed_objects[uid]['blender_output']}")
            # Blender bug process exits with error due to minor memory leak but object is converted successfully
            if (
                uid in failed_objects
                and "blender_output" in failed_objects[uid]
                and "Progress: 100.00%" in failed_objects[uid]["blender_output"]
            ):
                # Do not include this check because sometimes it fails regardless
                # and  "Error: Not freed memory blocks" in failed_objects[uid]['blender_output']:
                success = True

        if success and not args.skip_colliders:
            print("OBJ to colliders starting...")
            extra_args = {key: getattr(args, key) for key in extra_args_keys}
            start = time.perf_counter()
            success = obj_to_colliders(
                uid=uid,
                object_out_dir=asset_out_dir,
                max_colliders=max_colliders,
                capture_stdout=(not args.live),
                failed_objects=failed_objects,
                delete_objs=args.delete_objs,
                **extra_args,
            )
            end = time.perf_counter()
            print(f"OBJ to colliders success: {success}. Runtime: {end-start}s")

        print(f"=---- file {asset_out_dir} {uid} extension {args.extension}")
        # Save to desired format, compression step
        save_thor_asset_file(
            asset_json=add_default_annotations(
                load_existing_thor_asset_file(out_dir=asset_out_dir, object_name=uid),
                asset_directory=asset_out_dir,
            ),
            save_path=get_extension_save_path(
                out_dir=asset_out_dir, asset_id=uid, extension=args.extension
            ),
        )
        if args.extension != ".json" and not args.keep_json_asset:
            print("--- Removing .json asset")
            json_asset_path = get_existing_thor_asset_file_path(
                out_dir=asset_out_dir, asset_id=uid, force_extension=".json"
            )
            os.remove(json_asset_path)

        if success and not args.skip_thor_creation:
            import ai2thor.controller
            import ai2thor.fifo_server

            start = time.perf_counter()
            if not controller:
                controller = ai2thor.controller.Controller(
                    # local_build=True,
                    commit_id=THOR_COMMIT_ID,
                    platform=args.thor_platform,
                    start_unity=True,
                    scene="Procedural",
                    gridSize=0.25,
                    width=args.width,
                    height=args.height,
                    server_class=ai2thor.fifo_server.FifoServer,
                    antiAliasing=None if args.skip_thor_visualization else "fxaa",
                    quality="Very Low" if args.skip_thor_visualization else "Ultra",
                )
            print("THOR visualization starting...")
            success, asset_metadata = validate_in_thor(
                controller,
                asset_out_dir,
                uid,
                os.path.join(asset_out_dir, "images"),
                failed_objects=failed_objects,
                skip_images=args.skip_thor_visualization,
                skybox_color=args.skybox_color.split(","),
                load_file_in_unity=not args.send_asset_to_controller,
                extension=args.extension,
            )
            end = time.perf_counter()
            print(f"THOR visualization success: {success}. Runtime: {end-start}s")
            controller.reset(scene="Procedural")

            if args.add_visualize_thor_actions:
                add_visualize_thor_actions(asset_id=uid, asset_dir=asset_out_dir)

            metadata_output_file = os.path.join(asset_out_dir, out_metadata_filename)
            if success and asset_metadata:
                with open(metadata_output_file, "w") as f:
                    json.dump(asset_metadata, f, indent=2)

            end = time.perf_counter()
            print(
                f"Finished Object '{uid}' success: {success}. Object Runtime: {end-start_obj_time}s"
            )

    failed_json_str = json.dumps(failed_objects)

    if len(failed_objects):
        with open(report_out_path, "w") as f:
            f.write(failed_json_str)
    print(f"Failed objects: {failed_json_str}")
    end = time.perf_counter()
    print(f"Total Runtime: {end-start_process_time}s")


if __name__ == "__main__":
    main(sys.argv[1:])
