import argparse
import glob
import os
import sys
from importlib import import_module
from typing import Any, Dict, Optional, cast

import numpy as np
import objaverse
from ai2thor.controller import Controller

from objathor.annotation.gpt_from_views import get_initial_annotation
from objathor.annotation.objaverse_annotations_utils import (
    get_objaverse_home_annotations,
    get_objaverse_ref_categories,
)

from objathor.asset_conversion.util import get_blender_installation_path
from objathor.annotation.write import write_annotation

from objathor_blender.asset_conversion.pipeline_to_thor import optimize_assets_for_thor
from objathor_blender.annotation.render_glb_from_angles import render_glb_from_angles


def annotate_asset(
    uid: str,
    glb_path: str,
    output_dir: str,
    render_dir: str,
    render_angles=(0, 90, 180, 270),
    delete_blender_render_dir=False,
    allow_overwrite=False,
    **kwargs: Any,
) -> None:
    save_path = os.path.join(output_dir, f"annotations.json.gz")
    if os.path.exists(save_path) and not allow_overwrite:
        raise ValueError(
            f"Annotations already exist at {save_path} and allow_overwrite is False"
        )
    render_dir = os.path.join(output_dir, "blender_renders")
    os.makedirs(render_dir, exist_ok=True)
    try:
        render_glb_from_angles(
            glb_path=glb_path,
            save_dir=render_dir,
            angles=render_angles,
        )

        anno, urls = get_initial_annotation(
            uid,
            thumbnail_urls_cfg=dict(
                base_url=render_dir,
                view_indices=[str(float(angle)) for angle in render_angles],
                local_renders=True,
            ),
        )
        anno["pose_z_rot_angle"] = np.deg2rad(render_angles[anno["frontView"]])

        anno["scale"] = float(anno["height"]) / 100
        anno["z_axis_scale"] = True

        anno["uid"] = uid
        write_annotation(anno, save_path, **kwargs)
    finally:
        if delete_blender_render_dir:
            if os.path.exists(render_dir):
                for p in glob.glob(os.path.join(render_dir, "*.png")) + glob.glob(
                    os.path.join(render_dir, "*.jpg")
                ):
                    os.remove(p)

                os.rmdir(render_dir)


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
            "The output directory to write to (in particular, data will be saved to `<output>/<uid>/*`."
        ),
    )
    parser.add_argument(
        "--use_objaversehome",
        action="store_true",
        help="If annotations already exist for this asset in Objaverse-Home, use them instead of generating new ones.",
    )
    # parser.add_argument(
    #     "--local_render",
    #     action="store_true",
    #     help="Generate object views to be uploaded to GPT locally (requires blender)",
    # )
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
    skip_thor_creation: bool,
    skip_thor_visualization: bool,
    add_visualize_thor_actions: bool,
    width: int,
    height: int,
    skybox_color: str,
    save_as_pkl: bool,
    absolute_texture_paths: bool,
    extension: str,
    send_asset_to_controller: bool,
    blender_as_module: bool,
    blender_installation_path: str,
    thor_platform: str,
    keep_json_asset: bool,
    controller: Optional[Controller] = None,
) -> None:
    output_dir_with_uid = cast(str, os.path.join(output_dir, uid))
    os.makedirs(output_dir_with_uid, exist_ok=True)

    objaverse_uids = objaverse.load_uids()
    is_objaverse = uid in objaverse_uids

    if glb_path is None:
        assert is_objaverse, "If glb_path is not provided, uid must be an objaverse uid"
        glb_path = objaverse.load_objects([uid])[uid]

    def render_with_blender():
        blender_render_dir = os.path.join(output_dir_with_uid, "blender_renders")
        if len(glob.glob(os.path.join(blender_render_dir, "*"))) < 4:
            os.makedirs(blender_render_dir)
            render_glb_from_angles(
                glb_path=glb_path,
                save_dir=blender_render_dir,
                angles=(0, 90, 180, 270),
            )

    # ANNOTATION
    annotations_path = os.path.join(output_dir_with_uid, f"annotations.json.gz")
    if os.path.exists(annotations_path):
        print(f"Annotations already exist at {annotations_path}, will use these.")
        render_with_blender()
    else:
        if (
            is_objaverse
            and use_objaversehome
            and uid in get_objaverse_home_annotations()
        ):
            anno = get_objaverse_home_annotations()[uid]
            if "ref_category" not in anno:
                anno["ref_category"] = get_objaverse_ref_categories()[uid]
            write_annotation(anno, annotations_path)

            render_with_blender()
        else:
            annotate_asset(
                uid=uid,
                glb_path=glb_path,
                output_dir=output_dir_with_uid,
            )

    # OPTIMIZATION
    optimize_assets_for_thor(
        output_dir=output_dir,
        uid_to_glb_path={uid: glb_path},
        annotations_path=output_dir,
        max_colliders=max_colliders,
        skip_glb=False,
        blender_as_module=blender_as_module,
        extension=extension,
        thor_platform=thor_platform,
        blender_installation_path=blender_installation_path,
        live=False,
        save_as_pkl=save_as_pkl,
        absolute_texture_paths=absolute_texture_paths,
        delete_objs=delete_objs,
        keep_json_asset=keep_json_asset,
        skip_thor_creation=skip_thor_creation,
        width=width,
        height=height,
        skip_thor_visualization=skip_thor_visualization,
        skybox_color=tuple(map(int, skybox_color.split(","))),
        send_asset_to_controller=send_asset_to_controller,
        add_visualize_thor_actions=add_visualize_thor_actions,
        controller=controller,
    )

    # TODO: Should we use CLIP to validate that the assets still look like the blender renders? Example below
    # import clip
    # import torch
    # import numpy as np
    # from PIL import Image
    #
    # annotations = compress_json.load(annotations_path)
    # blender_render_path = os.path.join(output_dir_with_uid, "blender_renders", f"render_{-annotations['pose_z_rot_angle']:0.1f}.png")
    #
    # blender_img = np.array(Image.open(blender_render_path).convert("RGBA"), dtype=np.uint8)
    # blender_img[blender_img[:, :, 3] == 0] = 255
    # blender_img = blender_img[:,:,:3]
    #
    # thor_img = np.array(Image.open(os.path.join(output_dir_with_uid, "thor_renders", f"0_1_0_90.jpg")).convert("RGB"), dtype=np.uint8
    #
    # model, preprocess = clip.load("RN50", device="cpu")
    # model.eval()
    #
    # with torch.no_grad():
    #     blender_features = model.encode_image(preprocess(Image.fromarray(blender_img)).unsqueeze(0))
    #     thor_features = model.encode_image(preprocess(Image.fromarray(thor_img)).unsqueeze(0))
    #
    #     print(torch.cosine_similarity(blender_features, thor_features))


if __name__ == "__main__":
    args = parse_args()
    annotate_and_optimize_asset(
        uid=args.uid,
        glb_path=args.glb,
        output_dir=args.output,
        use_objaversehome=args.use_objaversehome,
        max_colliders=args.max_colliders,
        delete_objs=args.delete_objs,
        skip_thor_creation=args.skip_thor_creation,
        skip_thor_visualization=args.skip_thor_visualization,
        add_visualize_thor_actions=args.add_visualize_thor_actions,
        width=args.width,
        height=args.height,
        skybox_color=args.skybox_color,
        save_as_pkl=args.save_as_pkl,
        absolute_texture_paths=args.absolute_texture_paths,
        extension=args.extension,
        send_asset_to_controller=args.send_asset_to_controller,
        blender_as_module=args.blender_as_module,
        blender_installation_path=args.blender_installation_path,
        thor_platform=args.thor_platform,
        keep_json_asset=args.keep_json_asset,
    )
