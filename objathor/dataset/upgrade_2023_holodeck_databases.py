import copy
import os

import compress_json
import compress_pickle
import numpy as np
import objaverse
import tqdm
from ai2thor.controller import Controller
from ai2thor.fifo_server import FifoServer

from objathor.asset_conversion.asset_conversion_constants import EMPTY_HOUSE_JSON_PATH
from objathor.asset_conversion.util import make_single_object_house
from objathor.constants import THOR_COMMIT_ID


def get_object_metadata_in_empty_house(
    controller: Controller,
    asset_id: str,
):
    house = make_single_object_house(
        asset_id=asset_id,
        instance_id=asset_id,
        house_path=EMPTY_HOUSE_JSON_PATH,
        skybox_color=(255, 255, 255),
    )
    controller.reset(house)
    controller.step("BBoxDistance", objectId0=asset_id, objectId1=asset_id)
    obj = next(
        o
        for o in controller.last_event.metadata["objects"]
        if o["objectId"] == asset_id
    )
    return obj


if __name__ == "__main__":
    base_old_data_dir = os.path.expanduser("~/.objathor-assets/holodeck/2023_09_23/tmp")
    base_new_data_dir = os.path.expanduser("~/.objathor-assets/holodeck/2023_09_23")

    old_database = compress_json.load(
        os.path.join(base_old_data_dir, "objaverse_holodeck_database.json")
    )
    old_sbert = compress_pickle.load(
        os.path.join(
            base_old_data_dir, "objaverse_holodeck_description_features_sbert.pkl"
        )
    )
    old_clip = compress_pickle.load(
        os.path.join(base_old_data_dir, "objaverse_holodeck_features_clip_3.pkl")
    )
    old_clip = np.reshape(old_clip, (old_clip.shape[0] // 3, 3, 768))
    old_uids = list(old_database.keys())
    assert len(old_uids) == old_sbert.shape[0] == old_clip.shape[0]

    all_objaverse_uids = set(objaverse.load_uids())

    key_to_data_dict = {
        k: dict(
            uids=[],
            clip_features=[],
            sbert_features=[],
            annotations={},
        )
        for k in ["thor", "objaverse"]
    }

    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        scene="Procedural",
        gridSize=0.25,
        width=100,
        height=100,
        server_class=FifoServer,
        antiAliasing=None,
        quality="Very Low",
        makeAgentsVisible=False,
    )

    for uid, sbert_feature, clip_feature, obj_data in tqdm.tqdm(
        zip(old_uids, old_sbert, old_clip, old_database.values()), total=len(old_uids)
    ):
        assert uid == obj_data["annotations"]["uid"]

        if uid in all_objaverse_uids:
            data_dict = key_to_data_dict["objaverse"]
        else:
            data_dict = key_to_data_dict["thor"]

        data_dict["uids"].append(uid)
        data_dict["clip_features"].append(clip_feature)
        data_dict["sbert_features"].append(sbert_feature)

        ann = copy.deepcopy(obj_data["annotations"])

        am = copy.deepcopy(obj_data["assetMetadata"])

        if uid not in all_objaverse_uids:
            bbox_corners = np.array(
                get_object_metadata_in_empty_house(controller, uid)[
                    "axisAlignedBoundingBox"
                ]["cornerPoints"]
            )
        else:
            bbox_corners = np.array(
                obj_data["objectMetadata"]["axisAlignedBoundingBox"]["cornerPoints"]
            )

        mins = bbox_corners.min(0)
        maxes = bbox_corners.max(0)

        am["boundingBox"] = {
            "min": {"x": float(mins[0]), "y": float(mins[1]), "z": float(mins[2])},
            "max": {"x": float(maxes[0]), "y": float(maxes[1]), "z": float(maxes[2])},
        }

        ann["thor_metadata"] = {
            "assetMetadata": am,
        }

        data_dict["annotations"][uid] = ann

    for k in ["objaverse", "thor"]:
        save_dir = os.path.join(base_new_data_dir, f"{k}_object_data")
        data_dict = key_to_data_dict[k]

        uids = data_dict["uids"]
        cf = np.stack(data_dict["clip_features"], axis=0)
        sbertf = np.stack(data_dict["sbert_features"], axis=0)

        compress_pickle.dump(
            {
                "uids": uids,
                "img_features": cf,
            },
            os.path.join(save_dir, "clip_features.pkl"),
        )

        compress_pickle.dump(
            {
                "uids": uids,
                "text_features": sbertf,
            },
            os.path.join(save_dir, "sbert_features.pkl"),
        )

        compress_json.dump(
            data_dict["annotations"],
            os.path.join(save_dir, "annotations.json.gz"),
            json_kwargs={"indent": 2},
        )
