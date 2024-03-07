import fnmatch
import os
import tarfile

import torch.cuda
import tqdm

from objathor.dataset.aggregate_asset_annotations import prepare_annotations
from objathor.dataset.generate_holodeck_features import (
    generate_features,
    DEFAULT_DEVICE,
)


def filter_func(tarinfo):
    exclude_patterns = [
        "annotations.json.gz",
        "thor_metadata.json",
        ".DS_Store",
        "._*",
        ".Spotlight-V100",
        ".Trashes",
        ".fseventsd",
        ".VolumeIcon.icns",
        ".apdisk",
        ".TemporaryItems",
        "*.lock",
    ]
    for pattern in exclude_patterns:
        if tarinfo.name == pattern or fnmatch.fnmatch(tarinfo.name, pattern):
            return None
    return tarinfo


def create_tar_of_assets(assets_dir: str, save_dir: str):
    save_path = os.path.abspath(os.path.join(save_dir, "assets.tar"))

    if os.path.exists(save_path):
        raise FileExistsError(
            f"{save_path} already exists. Please remove it before creating a new tar file."
        )

    cur_dir = os.getcwd()
    try:
        os.chdir(assets_dir)  # Change to the directory where the assets are located
        with tarfile.open(save_path, "w") as tar:
            for root, dirs, files in tqdm.tqdm(os.walk(".")):
                for file in files:
                    file_path = os.path.join(root, file)
                    tar.add(file_path, filter=filter_func)
    except:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    finally:
        os.chdir(cur_dir)


def postprocess_assets(dataset_dir: str):
    assets_dir = os.path.join(dataset_dir, "assets")

    # Prepare annotations
    print("Preparing annotations...")
    prepare_annotations(save_dir=dataset_dir, assets_dir=assets_dir)

    # Create assets.tar
    print("Creating assets.tar...")
    create_tar_of_assets(assets_dir=assets_dir, save_dir=dataset_dir)

    # Generating holodeck features
    print("Generating holodeck features...")
    generate_features(
        base_dir=dataset_dir,
        annotations_path=os.path.join(dataset_dir, "annotations.json.gz"),
        device=DEFAULT_DEVICE,
        batch_size=32 if torch.cuda.is_available() else 8,
        num_workers=8,
    )
