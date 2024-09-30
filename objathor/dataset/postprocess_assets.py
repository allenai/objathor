import fnmatch
import os
import tarfile
from argparse import ArgumentParser

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


def create_tar_of_directory_with_exclusions(dir_to_tar: str, save_dir: str):
    dir_to_tar = os.path.abspath(dir_to_tar)
    save_path = os.path.abspath(
        os.path.join(save_dir, f"{os.path.basename(dir_to_tar)}.tar")
    )

    if os.path.exists(save_path):
        print(f"{save_path} already exists. Skipping...")
        return

    cur_dir = os.getcwd()
    try:
        os.chdir(dir_to_tar)  # Change to the directory where the assets are located
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


def postprocess_assets(dataset_dir: str, batch_size: int, num_workers: int):
    assets_dir = os.path.join(dataset_dir, "assets")

    # Prepare annotations
    print("Preparing annotations...")
    prepare_annotations(save_dir=dataset_dir, assets_dir=assets_dir)

    # Create assets.tar
    print("Creating assets.tar...")
    create_tar_of_directory_with_exclusions(dir_to_tar=assets_dir, save_dir=dataset_dir)

    # Generating holodeck features
    print("Generating holodeck features...")
    generate_features(
        base_dir=dataset_dir,
        asset_dir=assets_dir,
        annotations_path=os.path.join(dataset_dir, "annotations.json.gz"),
        device=DEFAULT_DEVICE,
        batch_size=batch_size if torch.cuda.is_available() else 8,
        num_workers=num_workers,
    )
    create_tar_of_directory_with_exclusions(
        dir_to_tar=os.path.join(dataset_dir, "features"), save_dir=dataset_dir
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Script to postprocess assets.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for datasets.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=(128 if torch.cuda.is_available() else 8),
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoader."
    )
    args = parser.parse_args()
    postprocess_assets(
        args.base_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
