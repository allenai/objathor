import gzip
import json
import os
import random
from typing import Optional

import ai2thor
import compress_json
import numpy as np
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import (
    ProceduralAssetHookRunner,
    get_all_asset_ids_recursively,
    create_assets_if_not_exist,
)
from matplotlib import pyplot as plt
from tqdm import tqdm

from objathor.constants import THOR_COMMIT_ID
from objathor.dataset import DatasetSaveConfig
from objathor.utils.download_utils import (
    download_with_progress_bar,
)


def read_jsonlgz(path: str, max_lines: Optional[int] = None):
    with gzip.open(path, "r") as f:
        lines = []
        for line in tqdm(f, desc=f"Loading {path}"):
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.last_asset_id_set = set()

    def Initialize(self, action, controller):
        if self.asset_limit > 0:
            return controller.step(
                action="DeleteLRUFromProceduralCache", assetLimit=self.asset_limit
            )

    def CreateHouse(self, action, controller):
        house = action["house"]
        asset_ids = get_all_asset_ids_recursively(house["objects"], [])
        asset_ids_set = set(asset_ids)
        if not asset_ids_set.issubset(self.last_asset_id_set):
            controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)
            self.last_asset_id_set = set(asset_ids)

        return create_assets_if_not_exist(
            controller=controller,
            asset_ids=asset_ids,
            asset_directory=self.asset_directory,
            copy_to_dir=None,
            asset_symlink=True,
            stop_if_fail=self.stop_if_fail,
            load_file_in_unity=self.load_file_in_unity,
            extension=self.extension,
            verbose=self.verbose,
        )


if __name__ == "__main__":
    SCENE_DATASET_DIR = os.path.join(
        os.path.expanduser("~/thor-scene-datasets/procthor_objaverse/2023_07_28/"),
    )
    os.makedirs(SCENE_DATASET_DIR, exist_ok=True)

    for split in ["train"]:
        download_with_progress_bar(
            f"https://pub-5932b61898254419952f5b13d42d82ab.r2.dev/procthor_objaverse%2F2023_07_28%2F{split}.jsonl.gz",
            save_path=os.path.join(
                SCENE_DATASET_DIR,
                f"{split}.jsonl.gz",
            ),
        )

    BASE_PATH = DatasetSaveConfig(VERSION="2023_07_28").VERSIONED_PATH
    BASE_ASSETS_DIR = os.path.join(BASE_PATH, "assets")
    BASE_ANNOTATIONS_PATH = os.path.join(BASE_PATH, "annotations.json.gz")

    ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
        asset_directory=BASE_ASSETS_DIR,
        asset_symlink=True,
        verbose=True,
        asset_limit=200,
    )

    controller = Controller(
        gridSize=0.2,  # Intentionally make this smaller than AGENT_MOVEMENT_CONSTANT to improve fidelity
        width=500,
        height=500,
        visibilityDistance=10,
        visibilityScheme="Distance",
        fieldOfView=90,
        server_class=ai2thor.fifo_server.FifoServer,
        useMassThreshold=True,
        massThreshold=10,
        autoSimulation=False,
        autoSyncTransforms=True,
        renderInstanceSegmentation=False,
        agentMode="stretch",
        renderDepthImage=False,
        cameraNearPlane=0.01,
        commit_id=THOR_COMMIT_ID,
        server_timeout=300,
        snapToGrid=False,
        fastActionEmit=True,
        action_hook_runner=ACTION_HOOK_RUNNER,
        antiAliasing="smaa",  # We can get nicer looking videos if we turn on antiAliasing and change the quality
        quality="Ultra",
    )

    train_scene_json_strs = read_jsonlgz(
        os.path.join(SCENE_DATASET_DIR, "train.jsonl.gz"), max_lines=10
    )
    annotations = compress_json.load(BASE_ANNOTATIONS_PATH)

    house = json.loads(train_scene_json_strs[0])

    controller.reset(house)
    agent = house["metadata"]["agent"]
    starting_pose = dict(
        position=agent["position"],
        rotation=agent["rotation"],
        horizon=agent["horizon"],
        standing=True,
    )
    controller.step("TeleportFull", **starting_pose, forceAction=True)

    for step_idx in range(1000):
        action = random.choice(["MoveAhead", "RotateRight", "RotateLeft", "MoveBack"])
        print(f"Step {step_idx}: taking action {action}")
        controller.step(action)
        print(f"Agent sees objaverse objects:")
        for o in controller.last_event.metadata["objects"]:
            if o["visible"] and o["assetId"] in annotations:
                ann = annotations[o["assetId"]]
                print(f'{ann["uid"]}: {ann["description"]}')

        plt.imshow(
            np.concatenate(
                (
                    controller.last_event.frame,
                    controller.last_event.third_party_camera_frames[0],
                ),
                axis=1,
            ),
        )
        plt.show()
