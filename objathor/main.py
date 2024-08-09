import argparse
import glob
import json
import os
import sys
import traceback
from functools import lru_cache
from importlib import import_module
from typing import Union, Callable, Any, Dict, Optional, cast, Literal, Tuple

import compress_json
import compress_pickle
import numpy as np
import objaverse
from PIL import Image
from ai2thor.controller import Controller

from objathor.annotation.annotation_utils import compute_llm_cost
from objathor.annotation.gpt_from_views import (
    get_initial_annotation,
    get_thumbnail_urls,
    get_gpt_dialogue_to_describe_asset_from_views,
    gpt_dialogue_to_batchable_request,
    load_gpt_annotations_from_json_str,
    get_gpt_dialogue_to_get_best_synset_using_annotations,
)
from objathor.annotation.objaverse_annotations_utils import (
    get_objaverse_home_annotations,
    get_objaverse_ref_categories,
    compute_clip_vit_l_similarity,
)
from objathor.annotation.synset_from_description import NUM_NEIGHS
from objathor.asset_conversion.pipeline_to_thor import optimize_assets_for_thor
from objathor.asset_conversion.util import get_blender_installation_path
from objathor.constants import OBJATHOR_CACHE_PATH, TEXT_LLM, VISION_LLM
from objathor.dataset.openai_batch_client import OpenAIBatchClient
from objathor.dataset.openai_batch_server import RequestStatus
from objathor.utils.blender import render_glb_from_angles
from objathor.utils.download_utils import download_with_locking


@lru_cache(maxsize=1)
def base_objaverse_annotations():
    return objaverse.load_annotations()


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


def write(
    anno: Dict[str, Any],
    output_file: Union[str, Callable[[Dict[str, Any]], Optional[Any]]],
    **kwargs: Any,
) -> None:
    if isinstance(output_file, str):
        if output_file.endswith(".json.gz"):
            compress_json.dump(anno, output_file, json_kwargs=dict(indent=2))
        elif output_file.endswith(".pickle.gz") or output_file.endswith(".pkl.gz"):
            compress_pickle.dump(anno, output_file)
        else:
            try:
                module_name, function_name = output_file.rsplit(".", 1)
                getattr(import_module(module_name), function_name)(anno, **kwargs)
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                print("Error", e)
                raise NotImplementedError(
                    "Only .pkl.gz and .json.gz supported, besides appropriate library function identifiers"
                )
    elif isinstance(output_file, Callable):
        output_file(anno)
    else:
        raise NotImplementedError(
            f"Unsupported output_file arg of type {type(output_file).__name__}"
        )


def annotate_asset(
    uid: str,
    glb_path: str,
    output_dir: str,
    render_angles=(0, 90, 180, 270),
    delete_blender_render_dir=False,
    allow_overwrite=False,
    extra_anns: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    save_path = os.path.join(output_dir, f"annotations.json.gz")
    if os.path.exists(save_path) and not allow_overwrite:
        raise ValueError(
            f"Annotations already exist at {save_path} and allow_overwrite is False"
        )
    render_dir = os.path.join(output_dir, "blender_renders")
    os.makedirs(render_dir, exist_ok=True)
    try:
        blender_render_paths = render_glb_from_angles(
            glb_path=glb_path,
            save_dir=render_dir,
            angles=render_angles,
        )
        for p in blender_render_paths:
            # Verify that images are not nearly completely white
            img = np.array(Image.open(p))
            if img.shape[-1] == 4:
                img[img[:, :, 3] == 0] = 255
                img = img[:, :, :3]

            if img.min() > 245:
                raise ValueError(
                    f"Image {p} is nearly completely white, likely a rendering error"
                )

        anno, urls = get_initial_annotation(
            uid,
            thumbnail_urls_cfg=dict(
                base_url=render_dir,
                view_indices=[str(float(angle)) for angle in render_angles],
                local_renders=True,
            ),
            get_best_synset=True,
        )
        anno["annotation_info"] = {"vision_llm": VISION_LLM, "text_llm": TEXT_LLM}

        # -1.0 * ... needed to undo the rotation of the object in the render
        anno["pose_z_rot_angle"] = -1.0 * np.deg2rad(render_angles[anno["frontView"]])

        anno["scale"] = float(anno["height"]) / 100
        anno["z_axis_scale"] = True

        anno["uid"] = uid

        if extra_anns is None:
            extra_anns = {}

        write(anno={**anno, **extra_anns}, output_file=save_path, **kwargs)
        return {**anno, **extra_anns}
    finally:
        if delete_blender_render_dir:
            if os.path.exists(render_dir):
                for p in glob.glob(os.path.join(render_dir, "*.png")) + glob.glob(
                    os.path.join(render_dir, "*.jpg")
                ):
                    os.remove(p)

                os.rmdir(render_dir)


def check_async_request_and_save_response_if_complete(
    request_uid: str, save_path: str, batch_client: OpenAIBatchClient
) -> RequestStatus:
    status = batch_client.check_status(uid=request_uid)

    if status == RequestStatus.COMPLETED:
        response = batch_client.get(uid=request_uid)

        compress_json.dump(obj=response, path=save_path, json_kwargs=dict(indent=2))
        _print_batch_cost_from_response(
            response, prefix=f"Request response saved to {save_path}. "
        )

        batch_client.delete(uid=request_uid)
        return status

    elif status in [
        RequestStatus.FAILED,
        RequestStatus.EXPIRED,
        RequestStatus.CANCELLED,
        RequestStatus.CANCELLING,
    ]:

        batch_client.delete(uid=request_uid)
        raise ValueError(f"Batch job failed with status {status}")
    else:
        return status


def _get_content_from_response(response: Dict[str, Any]) -> str:
    return response["response"]["body"]["choices"][0]["message"]["content"]


def _print_batch_cost_from_response(response: Dict[str, Any], prefix: str = "") -> None:
    try:
        usage = response["response"]["body"]["usage"]
        pt = usage["prompt_tokens"]
        ct = usage["completion_tokens"]
        model = response["response"]["body"]["model"]
        cost = (
            compute_llm_cost(input_tokens=pt, output_tokens=ct, model=model) * 0.5
        )  # 0.5 due to batching
        print(
            f"{prefix}Prompt tokens: {pt}."
            f" Completion tokens: {ct}."
            f" Approx cost: ${cost:.2g}.",
            flush=True,
        )
    except:
        print(f"{prefix}Failed to print cost from response", flush=True)


def async_annotate_asset(
    uid: str,
    glb_path: str,
    output_dir: str,
    async_host_and_port: str,
    render_angles=(0, 90, 180, 270),
    delete_blender_render_dir=False,
    allow_overwrite=False,
    extra_anns: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[RequestStatus, Optional[Dict[str, Any]]]:
    if allow_overwrite:
        raise NotImplementedError(
            "allow_overwrite=True is not supported for async_annotate_asset"
        )

    if delete_blender_render_dir:
        raise NotImplementedError(
            "delete_blender_render_dir=True is not supported for async_annotate_asset"
        )

    save_path = os.path.join(output_dir, f"annotations.json.gz")
    if os.path.exists(save_path) and not allow_overwrite:
        raise ValueError(
            f"Annotations already exist at {save_path} and allow_overwrite is False"
        )

    host, port_str = async_host_and_port.strip().split(":")
    batch_client = OpenAIBatchClient(host=host, port=int(port_str))

    annotate_from_views_uid_path = os.path.join(
        output_dir, f"annotate_from_views_uid.txt"
    )
    annotate_from_views_response_path = os.path.join(
        output_dir, "annotate_from_views_response.json"
    )

    synset_uid_path = os.path.join(output_dir, f"synset_uid.txt")
    synset_response_path = os.path.join(output_dir, "synset_response.json")

    render_dir = os.path.join(output_dir, "blender_renders")
    os.makedirs(render_dir, exist_ok=True)
    try:
        blender_render_paths = [
            path.replace("file://", "")
            for _, path in get_thumbnail_urls(
                uid,
                base_url=render_dir,
                view_indices=[str(float(angle)) for angle in render_angles],
                local_renders=True,
            )
        ]
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        blender_render_paths = render_glb_from_angles(
            glb_path=glb_path,
            save_dir=render_dir,
            angles=render_angles,
        )

    for p in blender_render_paths:
        # Verify that images are not nearly completely white
        img = np.array(Image.open(p))
        if img.shape[-1] == 4:
            img[img[:, :, 3] == 0] = 255
            img = img[:, :, :3]

        if img.min() > 245:
            raise ValueError(
                f"Image {p} is nearly completely white, likely a rendering error"
            )

    if not (os.path.exists(annotate_from_views_uid_path)):
        _, dialogue_dict = get_gpt_dialogue_to_describe_asset_from_views(
            uid,
            thumbnail_urls_cfg=dict(
                base_url=render_dir,
                view_indices=[str(float(angle)) for angle in render_angles],
                local_renders=True,
            ),
        )
        annotate_from_views_uid = batch_client.put(
            gpt_dialogue_to_batchable_request(gpt_dialogue=dialogue_dict)
        )
        if annotate_from_views_uid is None:
            raise RuntimeError(f"Failed to send annotate_from_views request for {uid}")

        with open(annotate_from_views_uid_path, "w") as f:
            f.write(annotate_from_views_uid)

    if not os.path.exists(annotate_from_views_response_path):
        with open(annotate_from_views_uid_path, "r") as f:
            annotate_from_views_uid = f.read().strip()

        status = check_async_request_and_save_response_if_complete(
            request_uid=annotate_from_views_uid,
            save_path=annotate_from_views_response_path,
            batch_client=batch_client,
        )
        if status != RequestStatus.COMPLETED:
            print(
                f"annotate_from_views request for {uid} is in state: {status}. Please re-run later if it is IN_PROGRESS or VALIDATING."
            )
            return status, None

    annotate_from_views_response = compress_json.load(annotate_from_views_response_path)
    json_str = _get_content_from_response(annotate_from_views_response)

    anno = load_gpt_annotations_from_json_str(
        uid=uid, json_str=json_str, attempt_cleanup=False
    )

    if not os.path.exists(synset_uid_path):
        dialogue_dict = get_gpt_dialogue_to_get_best_synset_using_annotations(
            annotation=anno,
            n_neighbors=2 * NUM_NEIGHS,
        )
        # Need to resave the json with the updated annotations as the near synsets have been added
        # by the above call
        annotate_from_views_response["response"]["body"]["choices"][0]["message"][
            "content"
        ] = json.dumps({"annotations": anno})
        compress_json.dump(
            annotate_from_views_response, annotate_from_views_response_path
        )

        synset_request = gpt_dialogue_to_batchable_request(gpt_dialogue=dialogue_dict)
        synset_uid = batch_client.put(synset_request)

        with open(synset_uid_path, "w") as f:
            f.write(synset_uid)

    if not os.path.exists(synset_response_path):
        with open(synset_uid_path, "r") as f:
            synset_uid = f.read().strip()

        status = check_async_request_and_save_response_if_complete(
            request_uid=synset_uid,
            save_path=synset_response_path,
            batch_client=batch_client,
        )
        if status != RequestStatus.COMPLETED:
            print(
                f"Synset request for {uid} is in state: {status}. Please re-run later if it is IN_PROGRESS or VALIDATING."
            )
            return status, None

    synset = _get_content_from_response(compress_json.load(synset_response_path))

    if synset.startswith("synset id: "):
        synset = synset.replace("synset id: ", "").strip()

    anno["synset"] = synset

    anno["annotation_info"] = {"vision_llm": VISION_LLM, "text_llm": TEXT_LLM}
    # -1.0 * ... needed to undo the rotation of the object in the render
    anno["pose_z_rot_angle"] = -1.0 * np.deg2rad(render_angles[anno["frontView"]])
    anno["scale"] = float(anno["height"]) / 100
    anno["z_axis_scale"] = True
    anno["uid"] = uid

    if extra_anns is None:
        extra_anns = {}

    write(anno={**anno, **extra_anns}, output_file=save_path, **kwargs)
    return RequestStatus.COMPLETED, {**anno, **extra_anns}


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
    width: int,
    height: int,
    skybox_color: str,
    absolute_texture_paths: bool,
    extension: Literal[".json", "json.gz", ".pkl.gz", ".msgpack", ".msgpack.gz"],
    send_asset_to_controller: bool,
    blender_as_module: bool,
    blender_installation_path: str,
    thor_platform: str,
    keep_json_asset: bool,
    compute_blender_thor_similarity: bool,
    controller: Optional[Controller] = None,
    async_host_and_port: Optional[str] = None,
) -> None:
    output_dir_with_uid = cast(str, os.path.abspath(os.path.join(output_dir, uid)))
    os.makedirs(output_dir_with_uid, exist_ok=True)

    print(f"Saving to {output_dir_with_uid}")

    objaverse_uids = objaverse.load_uids()
    is_objaverse = uid in objaverse_uids

    license_info = {}
    if is_objaverse:
        license_info["license_info"] = objaverse_license_info()[uid]

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
                blender_as_module=blender_as_module,
            )
            if len(glob.glob(os.path.join(blender_render_dir, "*"))) < 4:
                raise ValueError(
                    f"Failed to render the glb at {glb_path} with blender at {blender_render_dir}"
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
            write({**anno, **license_info}, annotations_path)

            render_with_blender()
        else:
            if async_host_and_port is not None:
                status, result = async_annotate_asset(
                    uid=uid,
                    glb_path=glb_path,
                    output_dir=output_dir_with_uid,
                    extra_anns=license_info,
                    async_host_and_port=async_host_and_port,
                )

                if status != RequestStatus.COMPLETED:
                    return

            else:
                annotate_asset(
                    uid=uid,
                    glb_path=glb_path,
                    output_dir=output_dir_with_uid,
                    extra_anns=license_info,
                )

    # OPTIMIZATION
    optimize_assets_for_thor(
        output_dir=output_dir,
        uid_to_glb_path={uid: glb_path},
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
        send_asset_to_controller=send_asset_to_controller,
        add_visualize_thor_actions=add_visualize_thor_actions,
        controller=controller,
    )

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
        send_asset_to_controller=args.send_asset_to_controller,
        blender_as_module=args.blender_as_module,
        blender_installation_path=args.blender_installation_path,
        thor_platform=args.thor_platform,
        keep_json_asset=args.keep_json_asset,
        compute_blender_thor_similarity=args.compute_similarity,
        async_host_and_port=args.async_host_and_port,
    )
