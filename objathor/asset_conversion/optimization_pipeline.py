import argparse
import glob
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from time import perf_counter
from typing import Any, List, Dict, Sequence, Optional, Union

import PIL.Image
import ai2thor.controller
import numpy as np
import objaverse

import objathor
from objathor.asset_conversion.colliders.generate_colliders import generate_colliders
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
from objathor.constants import ABS_PATH_OF_OBJATHOR, THOR_COMMIT_ID
from objathor.utils.types import (
    ObjathorStatus,
    PROCESSED_ASSET_EXTENSIONS,
    ObjathorInfo,
)

FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger(__name__)


@contextmanager
def Timer(s: str, pad_len: int = 70) -> float:
    s = (f"{{:<{pad_len}}}").format(s)
    print((f"{s}: starting").format(s), flush=True)
    start = perf_counter()
    yield None
    print(f"{s}: took {perf_counter() - start:.2f}s", flush=True)


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
    relative_texture_paths=True,
    run_blender_as_module=None,
    blender_installation_path=None,
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
            f"{blender_installation_path if blender_installation_path is not None else get_blender_installation_path()}"
            f" --background"
            f" --python {os.path.join(ABS_PATH_OF_OBJATHOR, 'asset_conversion', 'object_consolidater.py')}"
            f" --"
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

    command += (
        f' --object_path="{os.path.abspath(glb_path)}"'
        f' --output_dir="{os.path.abspath(object_out_dir)}"'
        f' --annotations="{annotations_path}"'
    )

    if generate_obj:
        command = command + " --obj"

    if relative_texture_paths:
        command = command + " --relative_texture_paths"

    if not capture_stdout:
        print(f"For {uid}, running command: {command}")

    process = None
    out = None
    timeout_hit = False
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
        timeout_hit = True
    except subprocess.CalledProcessError as e:
        result_code = e.returncode
        print(f"Blender call error: {out}\n{traceback.format_exc()}")
        out = f"{out}, Exception: {traceback.format_exc()}"
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        fmted_trace = traceback.format_exc()
        try:
            result_code = e.returncode
            out = f"{e.output}\n{fmted_trace}"
        except:
            result_code = -1
            out = f"Blender process error: {fmted_trace}"

        print(f"Blender process error: {traceback.format_exc()}")

    if not capture_stdout:
        print(f"Exited with code {result_code}")

    success = result_code == 0 and os.path.exists(
        os.path.join(object_out_dir, f"{uid}.obj")
    )

    if success:
        if not capture_stdout:
            print(f"---- Command ran successfully for {uid} at path {glb_path}")
    else:
        if "Progress: 100.00%" in out:
            # Blender bug process exits with error due to minor memory leak but object is converted successfully
            success = True
        else:
            if timeout_hit:
                failed_objects[uid][
                    "status"
                ] = ObjathorStatus.BLENDER_PROCESS_TIMEOUT_FAIL
            else:
                failed_objects[uid]["status"] = ObjathorStatus.BLENDER_PROCESS_FAIL

            failed_objects[uid]["blender_output"] = out if out else ""
            return False

    try:
        # The below compresses textures using the structural similarity metric
        # This is a lossy compression, but attempts to preserve the visual quality
        thor_obj_path = get_existing_thor_asset_file_path(object_out_dir, uid)

        # TODO: here optimize to remove needing to decompress and change references,
        # always export as json from blender pipeline and change to desired compression here
        asset_json = load_existing_thor_asset_file(os.path.abspath(object_out_dir), uid)

        save_dir = os.path.dirname(thor_obj_path)
        for k in [
            "albedo",
            "metallic_smoothness",
            "normal",
            "emission",
            "roughness",
            "metallic",
        ]:
            png_path = os.path.join(save_dir, f"{k}.png")
            jpg_path = os.path.join(save_dir, f"{k}.jpg")

            if k in ["roughness", "metallic"]:
                # We don't need these maps as we have `metallic_smoothness`
                if os.path.exists(png_path):
                    os.remove(png_path)

                continue

            input_path = png_path
            if k == "metallic_smoothness":
                # Don't want to convert metallic smoothness to jpg as this would destroy the smoothness
                # which is encoded in the alpha channel. Instead we move the smoothness to the B channel
                # (which isn't storing any relevant information) and then convert to jpg.
                img = np.array(
                    PIL.Image.open(input_path).convert("RGBA"), dtype=np.uint8
                )
                img[:, :, 1] = img[:, :, 0]
                img[:, :, 2] = img[:, :, 3]
                img[:, :, 3] = 255
                PIL.Image.fromarray(img).convert("RGB").save(jpg_path)
                input_path = jpg_path

            compress_image_to_ssim_threshold(
                input_path=input_path,
                output_path=jpg_path,
                threshold=0.95,
            )

            if k not in ["roughness", "metallic"]:
                os.remove(png_path)

                if k == "metallic_smoothness":
                    k = "metallicSmoothness"

                asset_json[f"{k}TexturePath"] = asset_json[f"{k}TexturePath"].replace(
                    ".png", ".jpg"
                )

        y_rot = compute_thor_rotation_to_obtain_min_bounding_box(
            asset_json["vertices"], max_deg_change=45, increments=91
        )
        asset_json["yRotOffset"] = y_rot
        print(f"Pose adjusted by {y_rot:.2f} degrees ({uid})")

        save_thor_asset_file(asset_json, thor_obj_path)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        logger.error(f"Exception: {e}")
        failed_objects[uid]["status"] = ObjathorStatus.IMAGE_COMPRESS_FAIL
        failed_objects[uid]["exception"] = traceback.format_exc()
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
            failed_objects[uid]["status"] = ObjathorStatus.GENERATE_COLLIDERS_FAIL
            failed_objects[uid]["generate_colliders_info"] = result_info[uid]
            return False
        else:
            return True

    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        print(f"Exception while running 'generate_colliders'\n{traceback.format_exc()}")

        failed_objects[uid]["status"] = ObjathorStatus.GENERATE_COLLIDERS_FAIL
        failed_objects[uid]["traceback"] = traceback.format_exc()
        return False


def validate_in_thor(
    controller: Any,
    asset_dir: Optional[str],
    asset_id: str,
    output_dir: str,
    failed_objects: OrderedDictWithDefault,
    skip_images=False,
    skybox_color=(255, 255, 255),
    load_file_in_unity=False,
    extension=None,
    angles: Union[int, Sequence[float]] = 90,
    axes=((0, 1, 0),),
):
    controller.reset(scene="Procedural")
    controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)

    evt = None
    try:
        if asset_dir is None:
            ad = controller.step(action="GetAssetDatabase").metadata["actionReturn"]
            assert asset_id in ad, f"Asset {asset_id} not in asset database"
            asset_metadata = ad[asset_id]
        else:
            evt = create_asset(
                thor_controller=controller,
                asset_directory=asset_dir,
                asset_id=asset_id,
                load_file_in_unity=load_file_in_unity,
                extension=extension,
            )
            if not evt.metadata["lastActionSuccess"]:
                failed_objects[asset_id] = {
                    "status": ObjathorStatus.THOR_CREATE_ASSET_FAIL,
                    "lastAction": controller.last_action,
                    "info": {
                        "asset_dir": asset_dir,
                        "asset_id": asset_id,
                        "extension": extension,
                    },
                    "errorMessage": evt.metadata["errorMessage"],
                }
                return False, None

            asset_metadata = evt.metadata["actionReturn"]

        if asset_metadata is None:
            failed_objects[asset_id] = {
                "status": ObjathorStatus.THOR_CREATE_ASSET_FAIL,
                "lastAction": controller.last_action,
                "info": {
                    "asset_dir": asset_dir,
                    "asset_id": asset_id,
                    "extension": extension,
                },
                "errorMessage": evt.metadata["errorMessage"],
            }
            return False, None

        if "objectMetadata" in asset_metadata:
            del asset_metadata["objectMetadata"]

        if not skip_images:
            if isinstance(angles, int):
                angles = [n * angles for n in range(0, round(360 / angles))]
            rotations = [(x, y, z, degrees) for degrees in angles for (x, y, z) in axes]

            evt = view_asset_in_thor(
                asset_id=asset_id,
                controller=controller,
                output_dir=output_dir,
                rotations=rotations,
                skybox_color=skybox_color,
            )
            if not evt.metadata["lastActionSuccess"]:
                failed_objects[asset_id] = {
                    "status": ObjathorStatus.THOR_VIEW_ASSET_FAIL,
                    "lastAction": controller.last_action,
                    "errorMessage": evt.metadata["errorMessage"],
                }
                return False, asset_metadata

        return True, asset_metadata
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        print(traceback.format_exc())
        failed_objects[asset_id] = {
            "status": ObjathorStatus.THOR_PROCESS_FAIL,
            "exception": traceback.format_exc(),
            "lastAction": controller.last_action,
            "errorMessage": evt.metadata["errorMessage"] if evt else "",
        }
        return False, None


def optimize_assets_for_thor(
    output_dir: str,
    uid: str,
    glb_path: str,
    annotations_path: str,
    max_colliders: int,
    blender_as_module: bool,
    extension: PROCESSED_ASSET_EXTENSIONS,
    skip_conversion: bool,
    skip_colliders: bool = False,
    skip_thor_metadata: bool = False,
    skip_thor_render: bool = False,
    thor_platform: Optional[str] = None,
    blender_installation_path: Optional[str] = None,
    controller: ai2thor.controller.Controller = None,
    live: bool = False,
    absolute_texture_paths: bool = False,
    delete_objs: bool = False,
    keep_json_asset: bool = False,
    width: int = 512,
    height: int = 512,
    skybox_color: Sequence[int] = (255, 255, 255),
    add_visualize_thor_actions: bool = False,
    log_prefix="",
    timeout: Optional[int] = None,
    overwrite: bool = False,
    **extra_collider_kwargs: Dict[str, Any],
) -> ObjathorInfo:
    if controller is not None:
        assert (width is None or controller.last_event.frame.shape[1] == width) and (
            height is None or controller.last_event.frame.shape[0] == height
        ), (
            f"Input height ({height}) or width ({width}) do not match the"
            f" input controller's frame shape ({controller.last_event.frame.shape})"
        )

    failed_objects = OrderedDictWithDefault(dict)

    start_process_time = time.perf_counter()

    given_controller = controller is not None
    try:
        start_obj_time = time.perf_counter()
        asset_out_dir = os.path.join(output_dir, uid)
        metadata_output_path = os.path.join(asset_out_dir, "thor_metadata.json")

        # First let's verify if we need to skip conversion, collider generation, or thor metadata generation
        optimized_object_save_path = get_extension_save_path(
            out_dir=asset_out_dir, asset_id=uid, extension=extension
        )
        if (not overwrite) and os.path.exists(optimized_object_save_path):
            asset = load_existing_thor_asset_file(
                out_dir=asset_out_dir, object_name=uid
            )

            has_colliders = "colliders" in asset
            has_valid_textures = True
            for key in asset.keys():
                if key.endswith("TexturePath"):
                    has_valid_textures = has_valid_textures and (
                        (os.path.isabs(asset[key]) and os.path.exists(asset[key]))
                        or os.path.exists(os.path.join(asset_out_dir, asset[key]))
                    )

            if has_colliders and has_valid_textures:
                print(
                    f"{log_prefix}{optimized_object_save_path} already exists and the asset has colliders."
                    f" Will skip conversion and collider generation."
                )
                skip_conversion = True
                skip_colliders = True
                if (
                    os.path.exists(metadata_output_path)
                    and os.stat(metadata_output_path).st_size > 0
                ):
                    if not skip_thor_metadata:
                        print(
                            f"{log_prefix}{metadata_output_path} already exists will skip"
                            f" generating thor metadata and thor images."
                        )
                        skip_thor_metadata = True
            else:
                raise RuntimeError(
                    f"{optimized_object_save_path} already exists but the asset does not have colliders or valid textures."
                    f" As overwrite is False, we cannot continue."
                )

        # Ensure annotations exist and determine their path
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

        # GLB to THOR conversion
        success = True
        if not skip_conversion:
            with Timer(f"{log_prefix}GLB to THOR ({uid})"):
                success = glb_to_thor(
                    glb_path=glb_path,
                    annotations_path=sub_annotations_path,
                    object_out_dir=asset_out_dir,
                    uid=uid,
                    failed_objects=failed_objects,
                    capture_stdout=not live,
                    generate_obj=True,
                    relative_texture_paths=not absolute_texture_paths,
                    run_blender_as_module=blender_as_module,
                    blender_installation_path=blender_installation_path,
                    timeout=timeout,
                )
            assert success == (uid not in failed_objects)

        # Collider generation
        if success and not skip_colliders:
            with Timer(f"{log_prefix}OBJ to collider ({uid})"):
                success = obj_to_colliders(
                    uid=uid,
                    object_out_dir=asset_out_dir,
                    max_colliders=max_colliders,
                    capture_stdout=(not live),
                    failed_objects=failed_objects,
                    delete_objs=delete_objs,
                    **{
                        "timeout": 60,
                        **extra_collider_kwargs,
                    },
                )
            assert success == (uid not in failed_objects)

        if success and (not skip_conversion) and (not skip_colliders):
            with Timer(
                f"{log_prefix}Saving {asset_out_dir} {uid} with extension {extension}"
            ):
                # Save to desired format, compression step
                save_path = get_extension_save_path(
                    out_dir=asset_out_dir, asset_id=uid, extension=extension
                )
                save_thor_asset_file(
                    asset_json=add_default_annotations(
                        load_existing_thor_asset_file(
                            out_dir=asset_out_dir, object_name=uid
                        ),
                        asset_directory=asset_out_dir,
                    ),
                    save_path=save_path,
                )
                if extension != ".json" and not keep_json_asset:
                    print(f"{log_prefix}Removing .json asset")
                    try:
                        json_asset_path = get_existing_thor_asset_file_path(
                            out_dir=asset_out_dir,
                            asset_id=uid,
                            force_extension=".json",
                        )
                        os.remove(json_asset_path)
                    except RuntimeError:
                        pass

                # Get size of GLB asset in MB
                glb_size = os.path.getsize(glb_path) / (1024 * 1024)
                # Get size of optimized asset in MB
                asset_size = os.path.getsize(save_path) / (1024 * 1024)
                for p in glob.glob(os.path.join(asset_out_dir, "*.jpg")):
                    asset_size += os.path.getsize(p) / (1024 * 1024)

                print(
                    f"{log_prefix}Original asset size {glb_size:.2f} MB,"
                    f" new asset size {asset_size:.2f}MB ({100 * (1 - asset_size / glb_size):0.2f}% reduction)",
                    flush=True,
                )

        # Validating the object can be spawned in thor, generate images in thor,
        # and save thor object metadata
        if success and not skip_thor_metadata:
            import ai2thor.controller
            import ai2thor.fifo_server

            with Timer(f"{log_prefix}THOR Metadata and visualization ({uid})"):
                if not controller:
                    controller = ai2thor.controller.Controller(
                        commit_id=THOR_COMMIT_ID,
                        fieldOfView=46,
                        platform=thor_platform,
                        start_unity=True,
                        scene="Procedural",
                        gridSize=0.25,
                        width=width,
                        height=height,
                        server_class=ai2thor.fifo_server.FifoServer,
                        antiAliasing=None if skip_thor_render else "fxaa",
                        quality="Very Low" if skip_thor_render else "Ultra",
                        makeAgentsVisible=False,
                    )

                controller.initialization_parameters["makeAgentsVisible"] = False
                success, asset_metadata = validate_in_thor(
                    controller=controller,
                    asset_dir=asset_out_dir,
                    asset_id=uid,
                    output_dir=os.path.join(asset_out_dir, "thor_renders"),
                    failed_objects=failed_objects,
                    skip_images=skip_thor_render,
                    skybox_color=skybox_color,
                    load_file_in_unity=True,
                    extension=extension,
                    angles=[0, 45, 90, 180, 270, 360 - 45],
                )
                assert success == (asset_metadata is not None)

                if success:
                    asset_metadata["thor_commit_id"] = THOR_COMMIT_ID
                    with open(metadata_output_path, "w") as f:
                        json.dump(asset_metadata, f, indent=2)

                if add_visualize_thor_actions:
                    objathor.asset_conversion.util.add_visualize_thor_actions(
                        asset_id=uid, asset_dir=asset_out_dir
                    )
            assert success == (uid not in failed_objects)

            end = time.perf_counter()
            print(
                f"{log_prefix}Finished Object '{uid}' success: {success}."
                f" Object Runtime: {end-start_obj_time:0.2f}s"
            )

        end = time.perf_counter()
        print(f"{log_prefix}Total Runtime: {end-start_process_time:0.2f}s")

    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        failed_objects[uid] = {
            "status": ObjathorStatus.UNKNOWN_OPTIMIZATION_FAIL,
            "exception": traceback.format_exc(),
        }
        print(traceback.format_exc())
    finally:
        try:
            if not given_controller:
                controller.stop()
        except:
            pass

    if uid in failed_objects:
        return failed_objects[uid]
    else:
        return {
            "status": ObjathorStatus.OPTIMIZATION_SUCCESS,
            "any_change": not (
                skip_conversion and skip_colliders and skip_thor_metadata
            ),
        }


def main(args):
    parser = argparse.ArgumentParser()
    print(f"Running pipeline_to_thor with args {args}")
    orig_args = args[:]

    parser.add_argument("--output_dir", type=str, default="./output", required=True)
    parser.add_argument(
        "--uids",
        type=str,
        default=None,
        help="Comma separated list of object ids (e.g. from objaverse) to process, overrides number. If this is"
        " unspecified, then we'll use the name of the glb file (see --glb_paths) as the uid (e.g. if the glb file is"
        " 'chair.glb' then the uid will be 'chair').",
    )
    parser.add_argument(
        "--glb_paths",
        type=str,
        default=None,
        help="Comma separated list of paths to .glb files (there should be one path per uid), if the uids passed"
        " are objaverse uids then this can be left unspecfied and we'll download the objaverse object.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="",
        help="Path to the annotations file, if it is a directory then we'll look for the annotations file.",
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
        "--delete_objs",
        action="store_true",
        help="Deletes objs after generating colliders.",
    )
    parser.add_argument(
        "--skip_conversion", action="store_true", help="Skips glb conversion."
    )
    parser.add_argument(
        "--skip_colliders",
        action="store_true",
        help="Skips obj to json collider generation.",
    )
    parser.add_argument(
        "--skip_thor_metadata",
        action="store_true",
        help="Skips THOR asset creation and visualization.",
    )
    parser.add_argument(
        "--skip_thor_render",
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
        "--absolute_texture_paths",
        action="store_true",
        help="Saves textures as absolute paths.",
    )

    parser.add_argument(
        "--extension",
        choices=[".json", "json.gz", ".pkl.gz", ".msgpack", ".msgpack.gz"],
        default=".json",
    )

    parser.add_argument(
        "--blender_as_module",
        action="store_true",
        help="Runs blender as a module. Requires bpy to be installed in python environment.",
    )

    parser.add_argument(
        "--keep_json_asset",
        action="store_true",
        help="Whether it keeps the intermediate .json asset file when using a non-json `extension`.",
    )

    found_blender = False
    try:
        get_blender_installation_path()
        found_blender = True
    except IOError:
        pass

    parser.add_argument(
        "--blender_installation_path",
        type=str,
        default=None,
        help="Blender installation path, when blender_as_module = False and"
        " we cannot find the installation path automatically.",
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

    args = parser.parse_args(orig_args)
    print(args)
    if args.verbose:
        # TODO use logger instead of print
        logger.setLevel(logging.DEBUG)

    uids = args.uids
    if uids is not None:
        uids = args.uids.split(",")

    glb_paths = args.glb_paths
    if glb_paths is not None:
        glb_paths = args.glb_paths.split(",")

    if uids is not None and glb_paths is not None:
        assert len(uids) == len(
            glb_paths
        ), "If uids and glb_paths are specified, then they must be the same length."
        uid_to_glb_path = {uid: glb_path for uid, glb_path in zip(uids, glb_paths)}
    elif uids is not None:
        uid_to_glb_path = objaverse.load_objects(
            uids=uids, download_processes=multiprocessing.cpu_count()
        )
    elif glb_paths is not None:
        uid_to_glb_path = {
            os.path.splitext(os.path.basename(glb_path))[0]: glb_path
            for glb_path in glb_paths
        }
    else:
        raise ValueError("Must specify either `uids` or `glb_paths`.")

    # noinspection PyTestUnpassedFixture
    optimize_assets_for_thor(
        output_dir=args.output_dir,
        uid_to_glb_path=uid_to_glb_path,
        annotations_path=args.annotations,
        max_colliders=args.max_colliders,
        skip_conversion=args.skip_conversion,
        skip_colliders=args.skip_colliders,
        skip_thor_metadata=args.skip_thor_metadata,
        skip_thor_render=args.skip_thor_render,
        blender_as_module=args.blender_as_module,
        extension=args.extension,
        thor_platform=args.thor_platform,
        blender_installation_path=args.blender_installation_path,
        live=args.live,
        absolute_texture_paths=args.absolute_texture_paths,
        delete_objs=args.delete_objs,
        keep_json_asset=args.keep_json_asset,
        width=args.width,
        height=args.height,
        skybox_color=tuple(map(int, args.skybox_color.split(","))),
        add_visualize_thor_actions=args.add_visualize_thor_actions,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
