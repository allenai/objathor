import argparse
import os
import sys
import traceback
from functools import lru_cache
from typing import Optional, cast

import compress_json
import numpy as np
import objaverse
from ai2thor.controller import Controller

from objathor.annotation.glb_to_annotation_pipeline import (
    NUM_BLENDER_RENDER_TRIES,
    write,
    annotate_asset,
    async_annotate_asset,
)
from objathor.annotation.objaverse_annotations_utils import (
    get_objaverse_home_annotations,
    get_objaverse_ref_categories,
    compute_clip_vit_l_similarity,
)
from objathor.asset_conversion.optimization_pipeline import optimize_assets_for_thor
from objathor.asset_conversion.util import get_blender_installation_path
from objathor.constants import OBJATHOR_CACHE_PATH
from objathor.utils.blender import render_glb_from_angles, BlenderRenderError
from objathor.utils.download_utils import download_with_locking
from objathor.utils.image_processing import verify_images_are_not_all_white
from objathor.utils.types import (
    PROCESSED_ASSET_EXTENSIONS,
    ObjathorStatus,
    ObjathorInfo,
)


@lru_cache(maxsize=1)
def objaverse_license_info():
    save_path = os.path.join(
        OBJATHOR_CACHE_PATH, "uid_to_objaverse_license_info.json.gz"
    )
    download_with_locking(
        url="https://pub-daedd7738a984186a00f2ab264d06a07.r2.dev/uid_to_objaverse_license_info.json.gz",
        save_path=save_path,
        lock_path=save_path + ".lock",
    )
    return compress_json.load(save_path)


def add_annotation_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--glb",
        type=str,
        default=None,
        help="Path to the .glb file of the asset to annotate",
    )
    parser.add_argument(
        "--uid",
        type=str,
        required=True,
        help="The UID of the asset, THIS MUST BE UNIQUE AMONG ALL ASSETS YOU INTEND TO PROCESS."
        " If this is an objaverse UID then the glb argument can be omitted"
        " as we'll attempt to download the asset from the objaverse database.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "The output directory to write to (in particular, data will be saved to `<output>/<uid>/*`)."
        ),
    )
    parser.add_argument(
        "--use_objaversehome",
        action="store_true",
        help="If annotations already exist for this asset in Objaverse-Home, use them instead of generating new ones.",
    )
    parser.add_argument(
        "--compute_similarity",
        action="store_true",
        help="Compute the cosine similarity between the blender and THOR renders and store it in the annotations.",
    )
    parser.add_argument(
        "--async_host_and_port",
        type=str,
        help="Host and port of the async server to use for annotation (a OpenAIBatchServer should have been launched on this host and port).",
    )
    return parser


def add_optimization_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
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
        "--width", type=int, default=512, help="Width of THOR asset visualization."
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of THOR asset visualization."
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
        help="Whether it keeps the intermediate .json asset file when storing in a different format to json.",
    )

    found_blender = False
    try:
        get_blender_installation_path()
        found_blender = True
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        pass

    parser.add_argument(
        "--blender_installation_path",
        type=str,
        default=None,
        help="Blender installation path, when blender_as_module = False and we cannot find the installation path automatically.",
        required=(not found_blender) and "--blender_as_module" not in sys.argv,
    )

    parser.add_argument(
        "--thor_platform",
        type=str,
        default="OSXIntel64" if sys.platform == "darwin" else "CloudRendering",
        help="THOR platform to use.",
        choices=["CloudRendering", "OSXIntel64"],
    )
    return parser


def parse_args(
    description="Generate GPT-based annotation of a 3D asset and optimize the asset for use in AI2-THOR.",
):
    parser = argparse.ArgumentParser(description=description)

    parser = add_annotation_arguments(parser)
    parser = add_optimization_arguments(parser)

    return parser.parse_args()


def annotate_and_optimize_asset(
    uid: Optional[str],
    glb_path: Optional[str],
    output_dir: str,
    use_objaversehome: bool,
    max_colliders: int,
    delete_objs: bool,
    skip_thor_metadata: bool,
    skip_thor_render: bool,
    add_visualize_thor_actions: bool,
    width: Optional[int],
    height: Optional[int],
    skybox_color: str,
    absolute_texture_paths: bool,
    extension: PROCESSED_ASSET_EXTENSIONS,
    blender_as_module: bool,
    thor_platform: str,
    keep_json_asset: bool,
    compute_blender_thor_similarity: bool,
    blender_installation_path: Optional[str] = None,
    controller: Optional[Controller] = None,
    async_host_and_port: Optional[str] = None,
) -> ObjathorInfo:
    if controller is not None:
        assert (width is None or controller.last_event.frame.shape[1] == width) and (
            height is None or controller.last_event.frame.shape[0] == height
        ), (
            f"Input height ({height}) or width ({width}) do not match the"
            f" input controller's frame shape ({controller.last_event.frame.shape})"
        )

    output_dir_with_uid = cast(str, os.path.abspath(os.path.join(output_dir, uid)))
    os.makedirs(output_dir_with_uid, exist_ok=True)

    print(f"Saving to {output_dir_with_uid}")

    objaverse_uids = objaverse.load_uids()
    is_objaverse = uid in objaverse_uids

    license_info = {}
    if is_objaverse:
        license_info["license_info"] = objaverse_license_info()[uid]

    if glb_path is None:
        assert uid is not None
        assert is_objaverse, "If glb_path is not provided, uid must be an objaverse uid"
        glb_path = objaverse.load_objects([uid])[uid]

    def render_with_blender():
        blender_render_dir = os.path.join(output_dir_with_uid, "blender_renders")
        os.makedirs(blender_render_dir, exist_ok=True)

        angles = (0, 90, 180, 270)

        blender_render_paths = []

        def enough_renders():
            return len(blender_render_paths) >= len(angles)

        for _ in range(NUM_BLENDER_RENDER_TRIES):
            if enough_renders():
                break

            try:
                blender_render_paths = render_glb_from_angles(
                    glb_path=glb_path,
                    save_dir=blender_render_dir,
                    angles=angles,
                    blender_as_module=blender_as_module,
                )
            except BlenderRenderError:
                blender_render_paths = []

            if len(blender_render_paths) != len(
                angles
            ) or not verify_images_are_not_all_white(blender_render_paths):
                return {"status": ObjathorStatus.BLENDER_RENDER_FAIL}

        if not enough_renders():
            raise BlenderRenderError(
                f"Failed to render the glb at {glb_path} with blender at {blender_render_dir}"
            )

    # ANNOTATION
    annotations_path = os.path.join(output_dir_with_uid, f"annotations.json.gz")
    annotations_path_no_gz = annotations_path[: -len(".gz")]

    if os.path.exists(annotations_path_no_gz) and not os.path.exists(annotations_path):
        compress_json.dump(
            compress_json.load(annotations_path_no_gz),
            annotations_path,
            json_kwargs=dict(indent=2),
        )

    if os.path.exists(annotations_path):
        print(f"Annotations already exist at {annotations_path}, will use these.")
        if render_with_blender() is None:
            return {"status": ObjathorStatus.BLENDER_RENDER_FAIL}
    else:
        if (
            is_objaverse
            and use_objaversehome
            and uid in get_objaverse_home_annotations()
        ):
            anno = get_objaverse_home_annotations()[uid]
            if "ref_category" not in anno:
                anno["ref_category"] = get_objaverse_ref_categories()[uid]
            write({**anno, **license_info}, annotations_path)

            if render_with_blender() is None:
                return {"status": ObjathorStatus.BLENDER_RENDER_FAIL}
        else:
            if async_host_and_port is not None:
                annotation_info = async_annotate_asset(
                    uid=uid,
                    glb_path=glb_path,
                    output_dir=output_dir_with_uid,
                    extra_anns=license_info,
                    async_host_and_port=async_host_and_port,
                )
            else:
                annotation_info = annotate_asset(
                    uid=uid,
                    glb_path=glb_path,
                    output_dir=output_dir_with_uid,
                    extra_anns=license_info,
                )

            if not annotation_info["status"].is_success():
                return annotation_info

    # OPTIMIZATION
    optimization_info = optimize_assets_for_thor(
        output_dir=output_dir,
        uid=uid,
        glb_path=glb_path,
        annotations_path=output_dir,
        max_colliders=max_colliders,
        skip_conversion=False,
        blender_as_module=blender_as_module,
        extension=extension,
        thor_platform=thor_platform,
        blender_installation_path=blender_installation_path,
        live=False,
        absolute_texture_paths=absolute_texture_paths,
        delete_objs=delete_objs,
        keep_json_asset=keep_json_asset,
        skip_thor_metadata=skip_thor_metadata,
        width=width,
        height=height,
        skip_thor_render=skip_thor_render,
        skybox_color=tuple(map(int, skybox_color.split(","))),
        add_visualize_thor_actions=add_visualize_thor_actions,
        controller=controller,
    )

    if not optimization_info["status"].is_success():
        return optimization_info

    if compute_blender_thor_similarity:
        annotations = compress_json.load(annotations_path)

        if "blender_thor_similarity" in annotations:
            print(
                f"Blender-THOR similarity already computed for {uid}, will not recompute."
            )
        else:
            thor_render_path = os.path.join(
                output_dir_with_uid, "thor_renders", f"0_1_0_0.0.jpg"
            )
            if not os.path.exists(thor_render_path):
                print(
                    f"THOR render does not exist at {thor_render_path}, will not compute similarity."
                )
            else:
                print("Computing similarity between blender and THOR renders.")
                try:
                    import torch

                    if controller is not None and controller.gpu_device is not None:
                        device = torch.device(controller.gpu_device)
                    else:
                        device = torch.device("cpu")

                    blender_render_path = os.path.join(
                        output_dir_with_uid,
                        "blender_renders",
                        f"render_{(-np.rad2deg(annotations['pose_z_rot_angle'])) % 360:0.1f}.jpg",
                    )

                    sim = compute_clip_vit_l_similarity(
                        img_path0=blender_render_path,
                        img_path1=thor_render_path,
                        device=device,
                    )
                    annotations["blender_thor_similarity"] = sim
                    compress_json.dump(
                        obj=annotations,
                        path=annotations_path,
                        json_kwargs=dict(indent=2),
                    )
                except (SystemExit, KeyboardInterrupt):
                    raise
                except:
                    print(
                        f"Failed to compute similarity, will not store in annotations, traceback:\n{traceback.format_exc()}"
                    )

    return {"status": ObjathorStatus.SUCCESS}


if __name__ == "__main__":
    args = parse_args()
    annotate_and_optimize_asset(
        uid=args.uid,
        glb_path=args.glb,
        output_dir=args.output,
        use_objaversehome=args.use_objaversehome,
        max_colliders=args.max_colliders,
        delete_objs=args.delete_objs,
        skip_thor_metadata=args.skip_thor_metadata,
        skip_thor_render=args.skip_thor_render,
        add_visualize_thor_actions=args.add_visualize_thor_actions,
        width=args.width,
        height=args.height,
        skybox_color=args.skybox_color,
        absolute_texture_paths=args.absolute_texture_paths,
        extension=args.extension,
        blender_as_module=args.blender_as_module,
        blender_installation_path=args.blender_installation_path,
        thor_platform=args.thor_platform,
        keep_json_asset=args.keep_json_asset,
        compute_blender_thor_similarity=args.compute_similarity,
        async_host_and_port=args.async_host_and_port,
    )
