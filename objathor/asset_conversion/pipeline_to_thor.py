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
from typing import Any, List, Dict, Sequence, Optional

import numpy as np
import objaverse

import objathor
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
import ai2thor.controller

FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger(__name__)


def save_asset_as(asset_id, asset_out_dir, extension, keep_json_asset=False):
    # Save to desired format, compression step
    save_thor_asset_file(
        asset_json=add_default_annotations(
            load_existing_thor_asset_file(out_dir=asset_out_dir, object_name=asset_id),
            asset_directory=asset_out_dir,
        ),
        save_path=get_extension_save_path(
            out_dir=asset_out_dir, asset_id=asset_id, extension=extension
        ),
    )
    if extension != ".json" and not keep_json_asset:
        json_asset_path = get_existing_thor_asset_file_path(
            out_dir=asset_out_dir, asset_id=asset_id, force_extension=".json"
        )
        os.remove(json_asset_path)


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
    run_blender_as_module=None,
    blender_instalation_path=None,
):
    os.makedirs(object_out_dir, exist_ok=True)

    if run_blender_as_module is None:
        try:
            import bpy

            run_blender_as_module = True
        except ImportError:
            run_blender_as_module = False
        logger.info(f"---- Autodetected run_blender_as_module={run_blender_as_module}")

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
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
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
    angle_increment: int = 90,
    axes=((0, 1, 0),),  # (1, 0, 0)),
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

        if not skip_images:
            angles = [
                n * angle_increment for n in range(0, round(360 / angle_increment))
            ]
            rotations = [(x, y, z, degrees) for degrees in angles for (x, y, z) in axes]
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


def optimize_assets_for_thor(
    output_dir: str,
    uid_to_glb_path: Dict[str, str],
    annotations_path: str,
    max_colliders: int,
    skip_glb: bool,
    blender_as_module: bool,
    extension: str,
    thor_platform: Optional[str] = None,
    blender_installation_path: Optional[str] = None,
    controller: ai2thor.controller.Controller = None,
    live: bool = False,
    save_as_pkl: bool = True,
    absolute_texture_paths: bool = False,
    delete_objs: bool = False,
    keep_json_asset: bool = False,
    skip_thor_creation: bool = False,
    width: int = 300,
    height: int = 300,
    skip_thor_visualization: bool = False,
    skybox_color: Sequence[int] = (255, 255, 255),
    send_asset_to_controller: bool = False,
    add_visualize_thor_actions: bool = False,
    skip_colliders: bool = False,
    **extra_collider_kwargs: Dict[str, Any],
) -> None:
    report_out_path = os.path.join(output_dir, "failed_objects.json")

    failed_objects = OrderedDictWithDefault(dict)

    print(f"--- Selected uids: {list(uid_to_glb_path.values())}")

    start_process_time = time.perf_counter()

    for uid, glb_path in uid_to_glb_path.items():
        start_obj_time = time.perf_counter()
        asset_out_dir = os.path.join(output_dir, uid)
        metadata_output_file = os.path.join(asset_out_dir, "thor_metadata.json")

        if os.path.isdir(annotations_path):
            if os.path.exists(os.path.join(annotations_path, f"{uid}.json")):
                sub_annotations_path = os.path.join(annotations_path, f"{uid}.json")
            elif os.path.exists(
                os.path.join(annotations_path, uid, f"annotations.json.gz")
            ):
                sub_annotations_path = os.path.join(
                    annotations_path, uid, f"annotations.json.gz"
                )
            else:
                raise RuntimeError(
                    f"Annotations path {annotations_path} does not contain annotations for {uid}"
                )
        else:
            sub_annotations_path = annotations_path

        success = True
        if not skip_glb:
            print("GLB to THOR starting...")
            start = time.perf_counter()
            success = glb_to_thor(
                glb_path=glb_path,
                annotations_path=sub_annotations_path,
                object_out_dir=asset_out_dir,
                uid=uid,
                failed_objects=failed_objects,
                capture_stdout=not live,
                generate_obj=True,
                save_as_json=not save_as_pkl,
                relative_texture_paths=not absolute_texture_paths,
                run_blender_as_module=blender_as_module,
                blender_instalation_path=blender_installation_path,
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

        if success and not skip_colliders:
            print("OBJ to colliders starting...")
            start = time.perf_counter()
            try:
                success = obj_to_colliders(
                    uid=uid,
                    object_out_dir=asset_out_dir,
                    max_colliders=max_colliders,
                    capture_stdout=(not live),
                    failed_objects=failed_objects,
                    delete_objs=delete_objs,
                    timeout=60,
                    **extra_collider_kwargs,
                )
            except subprocess.TimeoutExpired:
                print("OBJ to colliders timed out...")
                success = False

            end = time.perf_counter()

            print(f"OBJ to colliders success: {success}. Runtime: {end-start}s")

        if success:
            print(f"Saving {asset_out_dir} {uid} with extension {extension}")
            # Save to desired format, compression step
            save_thor_asset_file(
                asset_json=add_default_annotations(
                    load_existing_thor_asset_file(
                        out_dir=asset_out_dir, object_name=uid
                    ),
                    asset_directory=asset_out_dir,
                ),
                save_path=get_extension_save_path(
                    out_dir=asset_out_dir, asset_id=uid, extension=extension
                ),
            )
            if extension != ".json" and not keep_json_asset:
                print("Removing .json asset")
                json_asset_path = get_existing_thor_asset_file_path(
                    out_dir=asset_out_dir, asset_id=uid, force_extension=".json"
                )
                os.remove(json_asset_path)

        if success and not skip_thor_creation:
            import ai2thor.controller
            import ai2thor.fifo_server

            start = time.perf_counter()
            given_controller = controller is not None
            try:
                if not controller:
                    controller = ai2thor.controller.Controller(
                        # local_build=True,
                        commit_id=THOR_COMMIT_ID,
                        platform=thor_platform,
                        start_unity=True,
                        scene="Procedural",
                        gridSize=0.25,
                        width=width,
                        height=height,
                        server_class=ai2thor.fifo_server.FifoServer,
                        antiAliasing=None if skip_thor_visualization else "fxaa",
                        quality="Very Low" if skip_thor_visualization else "Ultra",
                    )

                print("THOR visualization starting...")
                success, asset_metadata = validate_in_thor(
                    controller,
                    asset_out_dir,
                    uid,
                    os.path.join(asset_out_dir, "thor_renders"),
                    failed_objects=failed_objects,
                    skip_images=skip_thor_visualization,
                    skybox_color=skybox_color,
                    load_file_in_unity=not send_asset_to_controller,
                    extension=extension,
                )
                end = time.perf_counter()
                print(f"THOR visualization success: {success}. Runtime: {end-start}s")
                controller.reset(scene="Procedural")

                if add_visualize_thor_actions:
                    objathor.asset_conversion.add_visualize_thor_actions(
                        asset_id=uid, asset_dir=asset_out_dir
                    )
            finally:
                try:
                    if not given_controller:
                        controller.stop()
                except:
                    pass

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


def main(args):
    parser = argparse.ArgumentParser()
    print("--------- argv")
    print(args)

    parser.add_argument("--output_dir", type=str, default="./output", required=True)
    parser.add_argument(
        "--uids",
        type=str,
        default="",
        help="Comma separated list of objaverse object ids to process, overrides number.",
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
        help="Whether it keeps the intermediate .json asset file when storing in a different format to json.",
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
        required=(not found_blender) and "--blender_as_module" not in args,
    )

    parser.add_argument(
        "--thor_platform",
        type=str,
        default="OSXIntel64" if sys.platform == "darwin" else "CloudRendering",
        help="THOR platform to use.",
        choices=["CloudRendering", "OSXIntel64"],  # Linux64
    )

    args, unknown = parser.parse_known_args(args)

    extra_args_keys = []
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            print(arg)
            extra_args_keys.append(arg.split("=")[0].removeprefix("--"))
            parser.add_argument(arg.split("=")[0], type=str)

    args = parser.parse_args(args)
    print(args)
    if args.verbose:
        # TODO use logger instead of print
        logger.setLevel(logging.DEBUG)

    uids = args.uids
    if uids != "":
        selected_uids = uids.split(",")
    else:
        with open(args.annotations, "r") as f:
            annotations = json.load(f)
        selected_uids = sorted(random.sample(annotations.keys(), args.number))

    uid_to_glb_path = objaverse.load_objects(
        uids=selected_uids, download_processes=multiprocessing.cpu_count()
    )

    optimize_assets_for_thor(
        output_dir=args.output_dir,
        uid_to_glb_path=uid_to_glb_path,
        annotations_path=args.annotations,
        max_colliders=args.max_colliders,
        skip_glb=args.skip_glb,
        blender_as_module=args.blender_as_module,
        extension=args.extension,
        thor_platform=args.thor_platform,
        blender_installation_path=args.blender_installation_path,
        live=args.live,
        save_as_pkl=args.save_as_pkl,
        absolute_texture_paths=args.absolute_texture_paths,
        delete_objs=args.delete_objs,
        keep_json_asset=args.keep_json_asset,
        skip_thor_creation=args.skip_thor_creation,
        width=args.width,
        height=args.height,
        skip_thor_visualization=args.skip_thor_visualization,
        skybox_color=tuple(map(int, args.skybox_color.split(","))),
        send_asset_to_controller=args.send_asset_to_controller,
        add_visualize_thor_actions=args.add_visualize_thor_actions,
        skip_colliders=args.skip_colliders,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
