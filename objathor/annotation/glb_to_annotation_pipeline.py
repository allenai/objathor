import glob
import json
import os
import traceback
from importlib import import_module
from json import JSONDecodeError
from typing import Dict, Any, Union, Callable, Optional

import compress_json
import compress_pickle
import numpy as np

from objathor.annotation.annotation_utils import compute_llm_cost
from objathor.annotation.gpt_from_views import (
    get_initial_annotation,
    get_gpt_dialogue_to_describe_asset_from_views,
    gpt_dialogue_to_batchable_request,
    load_gpt_annotations_from_json_str,
    get_gpt_dialogue_to_get_best_synset_using_annotations,
)
from objathor.annotation.synset_from_description import NUM_NEIGHS
from objathor.constants import VISION_LLM, TEXT_LLM
from objathor.dataset.openai_batch_client import OpenAIBatchClient
from objathor.dataset.openai_batch_constants import RequestStatus
from objathor.utils.blender import render_glb_from_angles, BlenderRenderError
from objathor.utils.image_processing import verify_images_are_not_all_white
from objathor.utils.types import (
    ObjathorInfo,
    ObjathorStatus,
)

NUM_BLENDER_RENDER_TRIES = 2


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
    overwrite=False,
    extra_anns: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ObjathorInfo:
    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not specified, cannot generate annotations."
        )

    save_path = os.path.join(output_dir, f"annotations.json.gz")
    if os.path.exists(save_path) and not overwrite:
        raise ValueError(
            f"Annotations already exist at {save_path} and overwrite is False"
        )

    render_dir = os.path.join(output_dir, "blender_renders")
    os.makedirs(render_dir, exist_ok=True)
    try:
        blender_error = ""
        try:
            blender_render_paths = render_glb_from_angles(
                glb_path=glb_path,
                save_dir=render_dir,
                angles=render_angles,
                overwrite=overwrite,
            )
        except BlenderRenderError:
            blender_error = traceback.format_exc()
            blender_render_paths = []

        if len(blender_render_paths) != len(
            render_angles
        ) or not verify_images_are_not_all_white(blender_render_paths):
            return {
                "status": ObjathorStatus.BLENDER_RENDER_FAIL,
                "exception": blender_error,
            }

        try:
            anno, urls = get_initial_annotation(
                uid,
                thumbnail_urls=list(enumerate(blender_render_paths)),
                get_best_synset=True,
            )
        except JSONDecodeError:
            return {"status": ObjathorStatus.JSON_DECODE_FAIL}

        anno["annotation_info"] = {"vision_llm": VISION_LLM, "text_llm": TEXT_LLM}

        # -1.0 * ... needed to undo the rotation of the object in the render
        anno["pose_z_rot_angle"] = -1.0 * np.deg2rad(render_angles[anno["frontView"]])

        anno["scale"] = float(anno["height"]) / 100
        anno["z_axis_scale"] = True

        anno["uid"] = uid

        if extra_anns is None:
            extra_anns = {}

        write(anno={**anno, **extra_anns}, output_file=save_path, **kwargs)
        return {
            "status": ObjathorStatus.ANNOTATION_SUCCESS,
            "annotations": {**anno, **extra_anns},
        }
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

    if status.is_complete():
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
    overwrite=False,
    extra_anns: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ObjathorInfo:
    if overwrite:
        raise NotImplementedError(
            "overwrite=True is not supported for async_annotate_asset"
        )

    if delete_blender_render_dir:
        raise NotImplementedError(
            "delete_blender_render_dir=True is not supported for async_annotate_asset"
        )

    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not specified, cannot generate annotations."
        )

    annotations_save_path = os.path.join(output_dir, f"annotations.json.gz")
    if os.path.exists(annotations_save_path) and not overwrite:
        raise ValueError(
            f"Annotations already exist at {annotations_save_path} and overwrite is False"
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

    blender_error = ""
    did_change = False
    try:
        blender_render_paths = render_glb_from_angles(
            glb_path=glb_path,
            save_dir=render_dir,
            angles=render_angles,
            overwrite=overwrite,
        )
    except BlenderRenderError:
        blender_error = traceback.format_exc()
        blender_render_paths = []

    if len(blender_render_paths) != len(
        render_angles
    ) or not verify_images_are_not_all_white(blender_render_paths):
        return {
            "status": ObjathorStatus.BLENDER_RENDER_FAIL,
            "exception": blender_error,
        }

    if not os.path.exists(annotate_from_views_uid_path):
        _, dialogue_dict = get_gpt_dialogue_to_describe_asset_from_views(
            uid,
            thumbnail_urls=list(enumerate(blender_render_paths)),
        )
        annotate_from_views_uid = batch_client.put(
            gpt_dialogue_to_batchable_request(gpt_dialogue=dialogue_dict)
        )
        if annotate_from_views_uid is None:
            raise RuntimeError(f"Failed to send annotate_from_views request for {uid}")

        with open(annotate_from_views_uid_path, "w") as f:
            f.write(annotate_from_views_uid)

        did_change = True

    if not os.path.exists(annotate_from_views_response_path):
        with open(annotate_from_views_uid_path, "r") as f:
            annotate_from_views_uid = f.read().strip()

        status = check_async_request_and_save_response_if_complete(
            request_uid=annotate_from_views_uid,
            save_path=annotate_from_views_response_path,
            batch_client=batch_client,
        )

        if status.is_fail():
            return {
                "status": ObjathorStatus.ASYNC_ANNOTATE_VIEWS_REQUEST_FAIL,
            }

        if status != RequestStatus.COMPLETED:
            print(
                f"annotate_from_views request for {uid} is in state:"
                f" {status}. Please re-run later if it is IN_PROGRESS or VALIDATING."
            )
            return {
                "status": ObjathorStatus.ANNOTATE_VIEWS_IN_PROGRESS,
                "did_change": did_change,
            }

        did_change = True

    annotate_from_views_response = compress_json.load(annotate_from_views_response_path)

    json_str = _get_content_from_response(annotate_from_views_response)

    try:
        anno = load_gpt_annotations_from_json_str(
            uid=uid, json_str=json_str, attempt_cleanup=False
        )
    except JSONDecodeError:
        return {"status": ObjathorStatus.JSON_DECODE_FAIL}

    if not os.path.exists(synset_uid_path):
        dialogue_dict = get_gpt_dialogue_to_get_best_synset_using_annotations(
            annotation=anno,
            n_neighbors=NUM_NEIGHS,
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

        did_change = True

    if not os.path.exists(synset_response_path):
        with open(synset_uid_path, "r") as f:
            synset_uid = f.read().strip()

        status = check_async_request_and_save_response_if_complete(
            request_uid=synset_uid,
            save_path=synset_response_path,
            batch_client=batch_client,
        )

        if status.is_fail():
            return {"status": ObjathorStatus.ASYNC_SYNSET_REQUEST_FAIL}

        if status != RequestStatus.COMPLETED:
            print(
                f"Synset request for {uid} is in state: {status}. Please re-run later if it is IN_PROGRESS or VALIDATING."
            )
            return {
                "status": ObjathorStatus.SYNSET_IN_PROGRESS,
                "did_change": did_change,
            }

        did_change = True

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

    anno = {**anno, **extra_anns}
    write(anno=anno, output_file=annotations_save_path, **kwargs)

    # Clean up the batch server request files
    for p in [
        annotate_from_views_uid_path,
        annotate_from_views_response_path,
        synset_uid_path,
        synset_response_path,
    ]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except (SystemExit, KeyboardInterrupt):
                raise
            except:
                pass

    return {
        "status": ObjathorStatus.ANNOTATION_SUCCESS,
        "annotations": anno,
    }
