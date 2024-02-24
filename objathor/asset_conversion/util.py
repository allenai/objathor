import json
import logging
import os
import pathlib
import shutil
from collections import OrderedDict
from io import BytesIO
from sys import platform
from typing import Tuple

import numpy as np

from objathor.asset_conversion.asset_conversion_constants import (
    EMPTY_HOUSE_JSON_PATH,
)  # DO NOT CHANGE THIS IMPORT, talk to Luca if something is broken for you

logger = logging.getLogger(__name__)

EXTENSIONS_LOADABLE_IN_UNITY = {
    ".json",
    ".json.gz",
    ".msgpack",
    ".msgpack.gz",
}


def compress_image_to_ssim_threshold(
    input_path: str,
    output_path: str,
    threshold: float,
    min_quality: int = 20,
    max_quality: int = 95,
) -> Tuple[int, float]:
    """
    Saves an image in the JPEG format at the lowest possible quality level without
    decreasing the Structural Similarity Index (SSIM) below a specified threshold.

    This function utilizes a binary search algorithm to find the optimal JPEG quality level
    between `min_quality` and `max_quality`. It ensures that the SSIM of the compressed image
    compared to the original does not fall below the provided `threshold`.
    The final image is saved at the determined quality level to the `output_path`.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path where the compressed image will be saved.
        threshold (float): The minimum acceptable SSIM value.
        min_quality (int, optional): The minimum quality level to consider for compression.
                                     Defaults to 20.
        max_quality (int, optional): The maximum quality level to consider for compression.
                                     Defaults to 95.

    Returns:
        tuple: A tuple containing two elements:
               - int: The quality level at which the image was saved.
               - float: The SSIM between the original and compressed image.

    Raises:
        AssertionError: If the input image is not in RGB format.

    Note:
        If the `threshold` cannot be reached at any quality level (i.e. it is too high) then the
        image is saved at `max_quality`.

    Example:
        >>> compress_image_to_ssim_threshold("path/to/original.png", "path/to/compressed.jpg", 0.9)
        (85, 0.912)
    """
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image

    # Load the image inside the function
    if input_path.lower().endswith(".jpg"):
        original_img = Image.open(input_path).convert("RGB")
    else:
        original_img_rgba = Image.open(input_path)
        original_img_rgba_on_white = Image.new("RGBA", original_img_rgba.size, "WHITE")
        original_img_rgba_on_white.paste(original_img_rgba, (0, 0), original_img_rgba)
        original_img = original_img_rgba_on_white.convert("RGB")

    original_img_np = np.array(original_img)
    assert original_img_np.shape[2] == 3
    left = min_quality  # Let's never go below this quality level
    right = max_quality  # Let's never go above this quality level
    # If we can't find a quality level that meets the threshold, we save at
    # 95 quality, not 100 because we never want to save at 100% quality regardless of what SSIM says (too big)
    best_quality = max_quality
    while left <= right:
        mid = (left + right) // 2
        with BytesIO() as f:
            original_img.save(f, "JPEG", quality=mid)
            f.seek(0)
            compressed_img = Image.open(f).convert("RGB")
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
        super().__init__()
        self.default_class = default_class

    def __missing__(self, key):
        value = self.default_class()
        self[key] = value
        return value


def get_msgpack_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.msgpack")


def get_msgpackgz_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.msgpack.gz")


def get_json_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.json")


def get_picklegz_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.pkl.gz")


def get_gz_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.gz")


def get_json_gz_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.json.gz")


def get_extension_save_path(out_dir, asset_id, extension):
    comp_extension = f".{extension}" if not extension.startswith(".") else extension
    return os.path.join(out_dir, f"{asset_id}{comp_extension}")


def get_existing_thor_asset_file_path(out_dir, asset_id, force_extension=None):
    possible_paths = OrderedDict(
        [
            (".json", get_json_save_path(out_dir, asset_id)),
            (".json.gz", get_json_gz_save_path(out_dir, asset_id)),
            (".msgpack", get_msgpack_save_path(out_dir, asset_id)),
            (".msgpack.gz", get_msgpackgz_save_path(out_dir, asset_id)),
            (".pkl.gz", get_picklegz_save_path(out_dir, asset_id)),
        ]
    )

    if force_extension is not None:
        if force_extension in possible_paths.keys():
            path = possible_paths[force_extension]
            if os.path.exists(path):
                return path
        else:
            raise Exception(
                f"Invalid extension `{force_extension}` for {asset_id}. Supported: {possible_paths.keys()}"
            )
    else:
        for path in possible_paths.values():
            if os.path.exists(path):
                return path

    raise RuntimeError(
        f"Could not find existing THOR object file for {asset_id}. Paths searched: {  possible_paths[force_extension] if force_extension else ', '.join(possible_paths.values())}"
    )


def load_existing_thor_asset_file(out_dir, object_name, force_extension=None):
    file_path = get_existing_thor_asset_file_path(
        out_dir, object_name, force_extension=force_extension
    )
    if file_path.endswith(".pkl.gz"):
        import compress_pickle

        return compress_pickle.load(file_path)
    elif file_path.endswith(".msgpack"):
        with open(file_path, "rb") as f:
            unp = f.read()
            import msgpack

            unp = msgpack.unpackb(unp)
            return unp
    elif file_path.endswith(".msgpack.gz"):
        import gzip

        with gzip.open(file_path, "rb") as f:
            unp = f.read()
            import msgpack

            unp = msgpack.unpackb(unp)
            return unp
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".json.gz"):
        import gzip

        with gzip.open(file_path, "rb") as f:
            unp = f.read()
            return json.dumps(unp)
    else:
        raise NotImplementedError(f"Unsupported file extension for path: {file_path}")


def load_existing_thor_metadata_file(out_dir):
    path = os.path.join(out_dir, f"thor_metadata.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def save_thor_asset_file(asset_json, save_path: str):
    extension = "".join(pathlib.Path(save_path).suffixes)
    if extension == ".msgpack.gz":
        import msgpack
        import gzip

        packed = msgpack.packb(asset_json)
        with gzip.open(save_path, "wb") as outfile:
            outfile.write(packed)
    elif extension == ".msgpack":
        import msgpack

        packed = msgpack.packb(asset_json)
        with open(save_path, "wb") as outfile:
            outfile.write(packed)
    elif extension in ["json.gz"]:
        import gzip

        with gzip.open(save_path, "wt") as outfile:
            json.dump(asset_json, outfile, indent=2)
    elif extension == ".pkl.gz":
        import compress_pickle

        compress_pickle.dump(
            obj=asset_json, path=save_path, pickler_kwargs={"protocol": 4}
        )
    elif extension.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(asset_json, f, indent=2)

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


def get_runtime_asset_filelock(save_dir, asset_id):
    return os.path.join(save_dir, f"{asset_id}.lock")


# TODO  remove  load_file_in_unity param
def create_runtime_asset_file(
    asset_directory,
    save_dir,
    asset_id,
    verbose=False,
    load_file_in_unity=False,
    use_extension=None,
):
    from filelock import FileLock

    # Verifies the file exists
    build_target_dir = os.path.join(save_dir, asset_id)
    asset = None

    # TODO figure out blender error
    with FileLock(get_runtime_asset_filelock(save_dir=save_dir, asset_id=asset_id)):
        exists = os.path.exists(build_target_dir)
        is_link = os.path.islink(build_target_dir)
        if exists and not is_link:
            # If not a link, delete the full directory
            if verbose:
                logger.info(f"Deleting old asset dir: {build_target_dir}")
            shutil.rmtree(build_target_dir)
        elif is_link:
            # If not a link, delete it only if its not pointing to the right place
            if os.path.realpath(build_target_dir) != os.path.realpath(asset_directory):
                os.remove(build_target_dir)

        if (not os.path.exists(build_target_dir)) and (
            not os.path.islink(build_target_dir)
        ):
            # Add symlink if it doesn't already exist
            print(f"Symlink from {asset_directory} to {build_target_dir}")
            os.symlink(
                os.path.abspath(asset_directory), os.path.abspath(build_target_dir)
            )

        if not load_file_in_unity:
            return load_existing_thor_asset_file(
                out_dir=build_target_dir, object_name=asset_id
            )
        return None


def change_asset_paths(asset, save_dir):
    asset["albedoTexturePath"] = os.path.join(
        save_dir,
        asset["name"],
        os.path.basename(asset["albedoTexturePath"]),
    )
    if "metallicSmoothnessTexturePath" in asset:
        asset["metallicSmoothnessTexturePath"] = os.path.join(
            save_dir,
            asset["name"],
            os.path.basename(asset["metallicSmoothnessTexturePath"]),
        )
    asset["normalTexturePath"] = os.path.join(
        save_dir,
        asset["name"],
        os.path.basename(asset["normalTexturePath"]),
    )
    if "emissionTexturePath" in asset:
        asset["emissionTexturePath"] = os.path.join(
            save_dir,
            asset["name"],
            os.path.basename(asset["emissionTexturePath"]),
        )
    return asset


def make_asset_pahts_relative(asset):
    return change_asset_paths(asset, ".")


def add_default_annotations(asset, asset_directory, verbose=False):
    thor_obj_md = load_existing_thor_metadata_file(out_dir=asset_directory)
    if thor_obj_md is None:
        if verbose:
            logger.info(f"Object metadata is missing annotations, assuming pickupable.")

        asset["annotations"] = {
            "objectType": "Undefined",
            "primaryProperty": "CanPickup",
            "secondaryProperties": (
                [] if asset.get("receptacleCandidate", False) else ["Receptacle"]
            ),
        }
    else:
        asset["annotations"] = {
            "objectType": "Undefined",
            "primaryProperty": thor_obj_md["assetMetadata"]["primaryProperty"],
            "secondaryProperties": thor_obj_md["assetMetadata"]["secondaryProperties"],
        }
    return asset


def create_asset(
    thor_controller,
    asset_id,
    asset_directory,
    copy_to_dir=None,
    verbose=False,
    load_file_in_unity=False,
    extension=None,
):
    # Verifies the file exists
    create_prefab_action = {}

    asset_path = get_existing_thor_asset_file_path(
        out_dir=asset_directory, asset_id=asset_id, force_extension=extension
    )
    file_extension = (
        "".join(pathlib.Path(asset_path).suffixes) if extension is None else extension
    )
    if file_extension not in EXTENSIONS_LOADABLE_IN_UNITY:
        load_file_in_unity = False

    copy_to_dir = (
        os.path.join(thor_controller._build.base_dir)
        if copy_to_dir is None
        else copy_to_dir
    )

    # save_dir = os.path.join(controller._build.base_dir, "processed_models")
    os.makedirs(copy_to_dir, exist_ok=True)

    if verbose:
        logger.info(f"Copying asset to THOR build dir: {copy_to_dir}.")

    asset = create_runtime_asset_file(
        asset_directory=asset_directory,
        save_dir=copy_to_dir,
        asset_id=asset_id,
        verbose=verbose,
        load_file_in_unity=load_file_in_unity,
    )

    if not load_file_in_unity:
        asset = change_asset_paths(asset=asset, save_dir=copy_to_dir)
        asset = add_default_annotations(
            asset=asset, asset_directory=asset_directory, verbose=verbose
        )
        create_prefab_action = {"action": "CreateRuntimeAsset", "asset": asset}
    else:
        create_prefab_action = {
            "action": "CreateRuntimeAsset",
            "id": asset_id,
            "dir": copy_to_dir,
            "extension": file_extension,
        }
        create_prefab_action = add_default_annotations(
            asset=create_prefab_action, asset_directory=asset_directory, verbose=verbose
        )

    evt = thor_controller.step(**create_prefab_action)
    print(f"Last Action: {thor_controller.last_action['action']}")
    if not evt.metadata["lastActionSuccess"]:
        logger.info(f"Last Action: {thor_controller.last_action['action']}")
        logger.info(f"Action success: {evt.metadata['lastActionSuccess']}")
        logger.info(f'Error: {evt.metadata["errorMessage"]}')

        logger.info(
            {
                k: v
                for k, v in create_prefab_action.items()
                if k
                in [
                    "action",
                    "name",
                    "receptacleCandidate",
                    "albedoTexturePath",
                    "metallicSmoothnessTexturePath",
                    "normalTexturePath",
                ]
            }
        )

    return evt


def make_single_object_house(
    asset_id,
    instance_id="asset_0",
    skybox_color=(0, 0, 0),
    house_path=EMPTY_HOUSE_JSON_PATH,
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
    rotations=tuple(),
    instance_id="asset_0",
    house_path=EMPTY_HOUSE_JSON_PATH,
    skybox_color=(255, 255, 255),
):
    from PIL import Image

    assert all(
        tuple(r[:3]) == (0, 1, 0) for r in rotations
    ), "Only rotations about z are supported."

    house = make_single_object_house(
        asset_id=asset_id,
        instance_id=instance_id,
        house_path=house_path,
        skybox_color=skybox_color,
    )
    controller.step(action="CreateHouse", house=house)
    controller.step(
        "Teleport", position={"x": 0, "y": 1000.0, "z": 0}, forceAction=True
    )

    angles = [r[-1] for r in rotations]

    frame_shape = controller.last_event.frame.shape
    controller.step("BBoxDistance", objectId0=instance_id, objectId1=instance_id)
    evt = controller.step(
        "RenderObjectFromAngles",
        objectId=instance_id,
        renderResolution={"x": frame_shape[1], "y": frame_shape[0]},
        angles=angles,
        cameraHeightMultiplier=0.5,
        raise_for_failure=True,
    )

    png_bytes_list = evt.metadata["actionReturn"]
    assert len(png_bytes_list) == len(angles)

    os.makedirs(output_dir, exist_ok=True)
    for png_bytes, rotation in zip(png_bytes_list, rotations):
        im = Image.open(BytesIO(png_bytes)).convert("RGB")
        im.save(
            os.path.join(
                output_dir,
                f"{rotation[0]}_{rotation[1]}_{rotation[2]}_{rotation[3]:0.1f}.jpg",
            )
        )

    return controller.last_event


def view_asset_in_thor_old(
    asset_id,
    controller,
    output_dir,
    rotations=tuple(),
    instance_id="asset_0",
    house_path=EMPTY_HOUSE_JSON_PATH,
    skybox_color=(255, 255, 255),
):
    from PIL import Image

    house = make_single_object_house(
        asset_id=asset_id,
        instance_id=instance_id,
        house_path=house_path,
        skybox_color=skybox_color,
    )
    evt = controller.step(action="CreateHouse", house=house)

    # Computing the bbox distance below causes the object-oriented bounding box to be created
    controller.step("BBoxDistance", objectId0=instance_id, objectId1=instance_id)
    obj = controller.step("AdvancePhysicsStep").metadata["objects"][1]
    obj_center_arr = np.array(obj["objectOrientedBoundingBox"]["cornerPoints"]).mean(0)

    if not evt.metadata["lastActionSuccess"]:
        print(f"Action success: {evt.metadata['lastActionSuccess']}")
        print(f'Error: {evt.metadata["errorMessage"]}')
        return evt

    controller.step(action="LookAtObjectCenter", objectId=instance_id)

    os.makedirs(output_dir, exist_ok=True)

    # Neural image if wanted, we don't really need this as the loop saves a neural position
    # image anyway
    # im = Image.fromarray(evt.frame)
    # im.save(os.path.join(output_dir, "neutral.jpg"))

    for rotation in rotations:
        controller.step(
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
        obj = controller.last_event.metadata["objects"][1]
        delta = obj_center_arr - np.array(
            obj["objectOrientedBoundingBox"]["cornerPoints"]
        ).mean(0)

        cur_pos = obj["position"]
        target_pos = {
            "x": cur_pos["x"] + delta[0],
            "y": cur_pos["y"] + delta[1],
            "z": cur_pos["z"] + delta[2],
        }

        controller.step(
            action="TeleportObject",
            objectId=instance_id,
            position=target_pos,
            rotation=obj["rotation"],
            forceAction=True,
            forceKinematic=True,
        )
        im = Image.fromarray(controller.last_event.frame)
        im.save(
            os.path.join(
                output_dir,
                f"{rotation[0]}_{rotation[1]}_{rotation[2]}_{rotation[3]}.jpg",
            )
        )
    return controller.last_event


def add_visualize_thor_actions(
    asset_id,
    asset_dir,
    instance_id="asset_0",
    house_path=EMPTY_HOUSE_JSON_PATH,
    house_skybox_color=(255, 255, 255),
):
    asset_json = os.path.join(asset_dir, f"{asset_id}.json")
    house = make_single_object_house(
        asset_id=asset_id,
        instance_id=instance_id,
        house_path=house_path,
        skybox_color=house_skybox_color,
    )
    actions = []
    with open(asset_json, "r") as f:
        asset = json.load(f)
        if isinstance(asset, dict):
            actions = [{"action": "CreateRuntimeAsset", "asset": asset}]
        elif not isinstance(asset, list):
            raise TypeError(
                f"Json {asset_json} is not a sequence of actions or a dictionary."
            )

    new_actions = [
        actions[0],
        dict(action="CreateHouse", house=house),
        dict(action="LookAtObjectCenter", objectId=instance_id),
    ]
    with open(asset_json, "w") as f:
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
