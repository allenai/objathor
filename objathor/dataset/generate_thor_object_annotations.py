import copy
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
import urllib.request
import warnings
from tempfile import TemporaryDirectory

import compress_json
import numpy as np
import openai
import setproctitle
from ai2thor.controller import Controller

from objathor.annotation.gpt_from_views import (
    DEFAULT_QUESTION_THOR_ASSET,
    get_initial_annotation,
)
from objathor.asset_conversion.pipeline_to_thor import validate_in_thor

# shared library
from objathor.asset_conversion.util import (
    OrderedDictWithDefault,
)
from objathor.constants import (
    THOR_COMMIT_ID,
    VISION_LLM,
    TEXT_LLM,
    ABS_PATH_OF_OBJATHOR,
)

mp = mp.get_context("spawn")

# Would prefer to use these old annotations, but they are misssing some assets
# OLD_THOR_ANNOTATIONS_PATH = os.path.join(
#     OBJATHOR_CACHE_PATH,
#     "holodeck",
#     "2023_09_23",
#     "thor_object_data",
#     "annotations.json.gz",
# )
# if not os.path.exists(OLD_THOR_ANNOTATIONS_PATH):
#     raise FileNotFoundError(
#         f"Could not find old thor annotations at {OLD_THOR_ANNOTATIONS_PATH}."
#         f" Please download the data using the download_holodeck_base_data.py script."
#     )
# OLD_THOR_ANNOTATIONS = compress_json.load(OLD_THOR_ANNOTATIONS_PATH)

_td = TemporaryDirectory()
_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

with _td as td_path:
    # Download old annotations file and load it
    _save_path = os.path.join(td_path, "procthor_database.json.gz")
    _req = urllib.request.Request(
        url="https://pub-daedd7738a984186a00f2ab264d06a07.r2.dev/misc/procthor_database_98aee1744dc634206a1f9a9b7bad04f8381acbe3.json.gz",
        headers=_headers,
    )
    with urllib.request.urlopen(_req) as response, open(_save_path, "wb") as _out_file:
        _data = response.read()  # a `bytes` object
        _out_file.write(_data)

    OLD_THOR_ANNOTATIONS = compress_json.load(_save_path)


def camel_case_to_words(s: str):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1 \2", s1).lower()


OBJECT_INFO_TEMPLATE = """\
To help you in your annotation, here's what we already know about this object:

Category: {category}
Mass: {mass:0.1g} kg
Height: {height:0.1g} cm
Width: {width:0.1g} cm
Depth: {depth:0.1g} cm
"""

OBJECT_INFO_TEMPLATE_NO_MASS = "\n".join(
    l for l in OBJECT_INFO_TEMPLATE.split("\n") if not l.lower().startswith("mass")
)


def annotate_procthor_asset(
    asset_id: str,
    controller: Controller,
    base_out_dir: str,
    failed_objects: OrderedDictWithDefault,
):
    print(f"Starting object '{asset_id}'")

    if asset_id not in OLD_THOR_ANNOTATIONS:
        warnings.warn(f"[ERROR] Object '{asset_id}' not found in old annotations")
        return

    asset_base_save_dir = os.path.join(base_out_dir, asset_id)

    annotations_save_path = os.path.join(asset_base_save_dir, "annotations.json.gz")
    metadata_save_path = os.path.join(asset_base_save_dir, "metadata.json")
    renders_save_dir = os.path.join(asset_base_save_dir, "thor_renders")

    if os.path.exists(annotations_save_path) and os.path.exists(metadata_save_path):
        print(f"Annotations and metadata already exist, skipping object '{asset_id}'")
        return

    start_obj_time = time.perf_counter()

    os.makedirs(asset_base_save_dir, exist_ok=True)

    success, asset_metadata = validate_in_thor(
        controller=controller,
        asset_dir=None,
        asset_id=asset_id,
        output_dir=renders_save_dir,
        failed_objects=failed_objects,
        skip_images=False,
        skybox_color=(255, 255, 255),
        load_file_in_unity=False,
        extension=None,
        angles=[0, 45, 90, 180, 270, 360 - 45],
    )
    assert success == (asset_metadata is not None)

    if not success:
        print(f"Failed to validate object '{asset_id}'")
        return

    asset_metadata["thor_commit_id"] = THOR_COMMIT_ID
    with open(metadata_save_path, "w") as f:
        json.dump(asset_metadata, f, indent=2)

    obj = next(
        obj
        for obj in controller.last_event.metadata["objects"]
        if obj["assetId"] == asset_id
    )

    category = camel_case_to_words(asset_metadata["objectType"])
    extra_info_kwargs = dict(
        mass=obj["mass"],
        category=category,
        height=100 * obj["axisAlignedBoundingBox"]["size"]["y"],
        width=100 * obj["axisAlignedBoundingBox"]["size"]["x"],
        depth=100 * obj["axisAlignedBoundingBox"]["size"]["z"],
    )

    if obj["mass"] > 0:
        question = DEFAULT_QUESTION_THOR_ASSET
        extra_user_info = OBJECT_INFO_TEMPLATE.format(**extra_info_kwargs)
    else:
        del extra_info_kwargs["mass"]
        question = DEFAULT_QUESTION_THOR_ASSET
        extra_user_info = OBJECT_INFO_TEMPLATE_NO_MASS.format(**extra_info_kwargs)

    annotations, _ = get_initial_annotation(
        uid=asset_id,
        thumbnail_urls=[
            (i, f"file://{renders_save_dir}/0_1_0_{float(angle):0.1f}.jpg")
            for i, angle in enumerate((0, 90, 180, 270))
        ],
        question=question,
        extra_user_info=extra_user_info,
        get_best_synset=False,
    )

    annotations["description_auto"] = f"a {category}"
    annotations.update(extra_info_kwargs)
    annotations["receptacle"] = obj["receptacle"]

    old_annotations = OLD_THOR_ANNOTATIONS[asset_id]["annotations"]
    for k in [
        "onCeiling",
        "onWall",
        "onFloor",
        "onObject",
    ]:
        annotations[k] = copy.deepcopy(old_annotations[k])

    annotations.update(
        {
            "location_annotated_by": "procthor",
            "source": "procthor",
            "size_annotated_by": "procthor",
        }
    )

    aabb = np.array(obj["axisAlignedBoundingBox"]["cornerPoints"])
    mins = dict(zip("xyz", aabb.min(0)))
    maxes = dict(zip("xyz", aabb.max(0)))

    annotations["thor_metadata"] = dict(
        assetMetadata=OLD_THOR_ANNOTATIONS[asset_id]["assetMetadata"]
    )
    annotations["thor_metadata"]["assetMetadata"]["boundingBox"] = {
        "min": mins,
        "max": maxes,
    }

    annotations["annotation_info"] = {"vision_llm": VISION_LLM, "text_llm": TEXT_LLM}

    annotations["scale"] = float(annotations["height"]) / 100
    annotations["z_axis_scale"] = True
    annotations["uid"] = asset_id

    compress_json.dump(annotations, annotations_save_path, json_kwargs={"indent": 2})

    end = time.perf_counter()
    print(
        f"Finished Object '{asset_id}' success: {success}. Object Runtime: {end - start_obj_time:0.2f}s"
    )


def generate_object_annotations_worker(
    worker_index: int, base_out_dir: str, asset_id_queue: mp.Queue
):
    setproctitle.setproctitle(title=f"Annotation worker {worker_index}")
    print(f"Starting annotation worker {worker_index}")

    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        fieldOfView=46,
        platform="OSXIntel64" if sys.platform == "darwin" else "CloudRendering",
        scene="Procedural",
        gridSize=0.25,
        width=512,
        height=512,
        antiAliasing="fxaa",
        quality="Ultra",
        makeAgentsVisible=False,
    )

    failed_objects = OrderedDictWithDefault(dict)

    while True:
        try:
            index, asset_id = asset_id_queue.get(timeout=1)
        except queue.Empty:
            break

        print(f"Worker {worker_index} processing object {asset_id} ({index})")
        try:
            annotate_procthor_asset(
                asset_id=asset_id,
                controller=controller,
                base_out_dir=base_out_dir,
                failed_objects=failed_objects,
            )
        except openai.OpenAIError:
            print(
                f"OpenAI API Error when processing object {asset_id}:\n{traceback.format_exc()}"
            )


if __name__ == "__main__":

    # Get the current date formatted as as YYYY_MM_DD
    # current_date = datetime.now()
    # formatted_date = current_date.strftime("%Y_%m_%d")
    formatted_date = "2024_08_05"

    base_out_dir = os.path.abspath(
        os.path.join(ABS_PATH_OF_OBJATHOR, "out", formatted_date, "thor_object_data")
    )
    os.makedirs(base_out_dir, exist_ok=True)

    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        platform="OSXIntel64" if sys.platform.lower() == "darwin" else "CloudRendering",
        scene="Procedural",
    )

    asset_database = controller.step("GetAssetDatabase").metadata["actionReturn"]

    q = mp.Queue()
    for i, aid in enumerate(asset_database.keys()):
        q.put((i, aid))

    print(f"Starting annotation generation for {len(asset_database)} objects")

    controller.stop()

    num_processes = 8
    processes = []
    for worker_index in range(num_processes):
        p = mp.Process(
            target=generate_object_annotations_worker,
            kwargs=dict(
                worker_index=worker_index, base_out_dir=base_out_dir, asset_id_queue=q
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
