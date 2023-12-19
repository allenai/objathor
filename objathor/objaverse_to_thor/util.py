import json
import os
import shutil
from collections import OrderedDict
from sys import platform

import numpy as np


def compress_image_to_ssim_threshold(
    input_path: str, output_path: str, threshold: float
):
    """Saves the image at the highest JPEG compression level that does not decrease
    the resulting SSIM beyond some threshold using binary search.
    """
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image

    # Load the image inside the function
    original_img = Image.open(input_path).convert("RGB")
    original_img_np = np.array(original_img)
    assert original_img_np.shape[2] == 3
    left = 1
    right = 100
    best_quality = 1
    while left <= right:
        mid = (left + right) // 2
        original_img.save(output_path, "JPEG", quality=mid)
        compressed_img = Image.open(output_path).convert("RGB")
        compressed_img_np = np.array(compressed_img)
        s = ssim(original_img_np, compressed_img_np, channel_axis=2)
        if s >= threshold:
            best_quality = mid
            right = mid - 1
        else:
            left = mid + 1
    # Save the image with the best quality found
    original_img.save(output_path, "JPEG", quality=best_quality)
    # Return the quality level and the SSIM
    return best_quality, s


class OrderedDictWithDefault(OrderedDict):
    def __init__(self, default_class):
        self.default_class = default_class

    def __missing__(self, key):
        value = self.default_class()
        self[key] = value
        return value


def get_json_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.json")


def get_picklegz_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.pkl.gz")


def get_existing_thor_obj_file_path(out_dir, object_name):
    possible_paths = [
        get_json_save_path(out_dir, object_name),
        get_picklegz_save_path(out_dir, object_name),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise Exception(f"Could not find existing THOR object file for {object_name}")


def load_existing_thor_obj_file(out_dir, object_name):
    path = get_existing_thor_obj_file_path(out_dir, object_name)
    if path.endswith(".pkl.gz"):
        import compress_pickle

        return compress_pickle.load(path)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise NotImplementedError(f"Unsupported file extension for path: {path}")


def save_thor_obj_file(data, save_path: str):
    if save_path.endswith(".pkl.gz"):
        import compress_pickle

        compress_pickle.dump(obj=data, path=save_path, pickler_kwargs={"protocol": 4})
    elif save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise NotImplementedError(
            f"Unsupported file extension for save path: {save_path}"
        )


def get_blender_installation_path():
    paths = {
        "darwin": [
            "/Applications/Blender.app/Contents/MacOS/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ],
        "linux": [
            # TODO: Add docker path
            os.path.join(os.path.expanduser("~"), "blender-3.2.2-linux-x64/blender"),
            os.path.join(os.getcwd(), "blender-3.2.2-linux-x64/blender"),
        ],
    }
    paths["linux2"] = paths["linux"]
    if platform in paths:
        for path in paths[platform]:
            if os.path.exists(path):
                return path
        raise Exception("Blender not found.")
    else:
        raise Exception(f'Unsupported platform "{platform}"')


def create_asset_in_thor(
    controller, uid, asset_directory, asset_symlink=True, verbose=False
):
    # Verifies the file exists
    get_existing_thor_obj_file_path(out_dir=asset_directory, object_name=uid)

    if verbose:
        print(
            f"Copying asset to Thor Build dir: {controller._build.base_dir} tmp: {controller._build.tmp_dir}"
        )

    if asset_symlink:
        build_target_dir = os.path.join(controller._build.base_dir, uid)
        if os.path.exists(build_target_dir):
            if not os.path.islink(build_target_dir):
                if verbose:
                    print(f"--- deleting old {build_target_dir}")
                shutil.rmtree(build_target_dir)
            else:
                tmp_symlink = os.path.join(controller._build.base_dir, "tmp")
                os.symlink(asset_directory, tmp_symlink)
                os.replace(tmp_symlink, build_target_dir)
        else:
            os.symlink(asset_directory, build_target_dir)
    else:
        build_target_dir = os.path.join(controller._build.base_dir, uid)

        if verbose:
            print("Starting copy and reference modification...")
        if os.path.exists(build_target_dir):
            if verbose:
                print(f"--- deleting old {build_target_dir}")
            shutil.rmtree(build_target_dir)

        if os.path.isabs(asset_directory):
            # TODO change json texures content
            asset_json_actions = load_existing_thor_obj_file(
                out_dir=asset_directory, object_name=uid
            )

            asset_json_actions["albedoTexturePath"] = os.path.join(
                uid, os.path.basename(asset_json_actions["albedoTexturePath"])
            )
            asset_json_actions["normalTexturePath"] = os.path.join(
                uid, os.path.basename(asset_json_actions["normalTexturePath"])
            )
            asset_json_actions["emissionTexturePath"] = os.path.join(
                uid, os.path.basename(asset_json_actions["emissionTexturePath"])
            )

            save_thor_obj_file(
                data=asset_json_actions,
                save_path=get_existing_thor_obj_file_path(
                    out_dir=build_target_dir, object_name=uid
                ),
            )

            if verbose:
                print("Reference modification finished.")

        shutil.copytree(
            asset_directory,
            build_target_dir,
            ignore=shutil.ignore_patterns("images", "*.obj", "thor_metadata.json"),
        )

        if verbose:
            print("Copy finished.")

    if verbose:
        print("After copy tree")

    create_prefab_action = load_existing_thor_obj_file(
        out_dir=asset_directory, object_name=uid
    )
    evt = controller.step(**create_prefab_action)

    if not evt.metadata["lastActionSuccess"]:
        print(f"Action success: {evt.metadata['lastActionSuccess']}")
        print(f'Error: {evt.metadata["errorMessage"]}')

        print(
            {
                k: v
                for k, v in create_prefab_action.items()
                if k
                in [
                    "action",
                    "name",
                    "receptacleCandidate",
                    "albedoTexturePath",
                    "normalTexturePath",
                    "emissionTexturePath",
                ]
            }
        )

    return evt


def make_single_object_house(
    asset_id,
    instance_id="asset_0",
    skybox_color=(0, 0, 0),
    house_path="./objaverse_to_thor/data/empty_house.json",
):
    with open(house_path, "r") as f:
        house = json.load(f)

    house["objects"] = [
        {
            "assetId": asset_id,
            "id": instance_id,
            "kinematic": True,
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "layer": "Procedural2",
            "material": None,
        }
    ]
    house["proceduralParameters"]["skyboxColor"] = {
        "r": skybox_color[0],
        "g": skybox_color[1],
        "b": skybox_color[2],
    }
    return house


def view_asset_in_thor(
    asset_id,
    controller,
    output_dir,
    rotations=[],
    instance_id="asset_0",
    house_path="./objaverse_to_thor/data/empty_house.json",
    skybox_color=(0, 0, 0),
):
    from PIL import Image

    house = make_single_object_house(
        asset_id=asset_id,
        instance_id=instance_id,
        house_path=house_path,
        skybox_color=skybox_color,
    )
    evt = controller.step(action="CreateHouse", house=house)

    if not evt.metadata["lastActionSuccess"]:
        print(f"Action success: {evt.metadata['lastActionSuccess']}")
        print(f'Error: {evt.metadata["errorMessage"]}')
        return evt
    evt = controller.step(action="LookAtObjectCenter", objectId=instance_id)

    im = Image.fromarray(evt.frame)
    # os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im.save(os.path.join(output_dir, "neutral.jpg"))
    for rotation in rotations:
        evt = controller.step(
            action="RotateObject",
            angleAxisRotation={
                "axis": {
                    "x": rotation[0],
                    "y": rotation[1],
                    "z": rotation[2],
                },
                "degrees": rotation[3],
            },
        )
        im = Image.fromarray(evt.frame)
        im.save(
            os.path.join(
                output_dir,
                f"{rotation[0]}_{rotation[1]}_{rotation[2]}_{rotation[3]}.jpg",
            )
        )
    return evt


def add_visualize_thor_actions(
    asset_id,
    asset_dir,
    instance_id="asset_0",
    house_path="./objaverse_to_thor/data/empty_house.json",
    house_skybox_color=(0, 0, 0),
):
    # asset_id = os.path.splitext(os.path.basename(output_json))[0]
    actions_json = os.path.join(asset_dir, f"{asset_id}.json")
    house = make_single_object_house(
        asset_id=asset_id,
        instance_id=instance_id,
        house_path=house_path,
        skybox_color=house_skybox_color,
    )

    with open(actions_json, "r") as f:
        actions = json.load(f)
        if isinstance(actions, dict):
            actions = [actions]
        if not isinstance(actions, list):
            raise TypeError(
                f"Json {actions_json} is not a sequence of actions or a dictionary."
            )

    new_actions = [
        actions[0],
        dict(action="CreateHouse", house=house),
        dict(action="LookAtObjectCenter", objectId=instance_id),
    ]
    with open(actions_json, "w") as f:
        json.dump(new_actions, f, indent=2)


def get_receptacle_object_types():
    return {
        "Sink",
        "Cart",
        "Toilet",
        "ToiletPaperHanger",
        "TowelHolder",
        "HandTowelHolder",
        "Bed",
        "Chair",
        "DiningTable",
        "Dresser",
        "Safe",
        "ShelvingUnit",
        "SideTable",
        "Stool",
        "Bowl",
        "CoffeeMachine",
        "Cup",
        "Fridge",
        "GarbageCan",
        "Microwave",
        "Mug",
        "Pan",
        "Plate",
        "Pot",
        "Toaster",
        "Box",
        "CoffeeTable",
        "Desk",
        "ArmChair",
        "Ottoman",
        "Sofa",
        "TVStand",
        "ClothesDryer",
        "CounterTop",
        "WashingMachine",
        "DogBed",
        "Footstool",
        "LaundryHamper",
    }
