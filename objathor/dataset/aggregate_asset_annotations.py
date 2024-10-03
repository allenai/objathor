import gzip
import json
import os
import tarfile
import warnings
from argparse import ArgumentParser
from typing import Dict, Any, Sequence, Optional

import compress_json
import tqdm
import concurrent.futures


def wait_for_futures_and_raise_errors(
    futures: Sequence[concurrent.futures.Future],
) -> Sequence[Any]:
    results = []
    concurrent.futures.wait(futures)
    for future in futures:
        try:
            results.append(future.result())  # This will re-raise any exceptions
        except Exception:
            raise
    return results


def add_thor_metadata_to_existing_annotations(annotations_path: str, assets_dir: str):
    annotations = compress_json.load(annotations_path)

    fails = 0
    for id, ann in tqdm.tqdm(annotations.items()):
        thor_metadata_path = os.path.join(assets_dir, id, "thor_metadata.json")
        if os.path.exists(thor_metadata_path):
            ann["thor_metadata"] = compress_json.load(thor_metadata_path)
            del ann["thor_metadata"]["objectMetadata"]
        else:
            fails += 1
            print(
                f"Warning: No THOR metadata found for {id} at path {thor_metadata_path}."
            )

    compress_json.dump(annotations, annotations_path, json_kwargs={"indent": 2})


def update_annotations(
    dir_or_tar_path: str,
    all_annotations: Dict[str, Any],
    pbar: Optional[tqdm.tqdm] = None,
):
    try:

        def print_or_log(msg):
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

        if not os.path.isdir(dir_or_tar_path) and not tarfile.is_tarfile(
            dir_or_tar_path
        ):
            print_or_log(f"WARNING: Invalid path: {dir_or_tar_path}")
            return

        uid = os.path.basename(dir_or_tar_path).replace(".tar", "")

        if uid in all_annotations:
            print_or_log(f"Annotations for {uid} already exist, skipping...")
            return

        print_or_log(f"Loading annotations for {uid}...")

        thor_metadata = None
        if os.path.isdir(dir_or_tar_path):
            annotations_path = os.path.join(dir_or_tar_path, "annotations.json.gz")
            thor_metadata_path = os.path.join(dir_or_tar_path, "thor_metadata.json")

            if not os.path.exists(annotations_path):
                print_or_log(
                    f"WARNING: Annotations not found at {annotations_path}, skipping..."
                )

            asset_ann = compress_json.load(annotations_path)
            try:
                thor_metadata = compress_json.load(thor_metadata_path)
            except:
                pass
        else:
            with tarfile.open(dir_or_tar_path, "r:*") as asset_tf:
                try:
                    with asset_tf.extractfile(f"{uid}/annotations.json.gz") as f:
                        asset_ann = json.loads(gzip.decompress(f.read()))
                except:
                    print_or_log(
                        f"WARNING: Annotations not found for {uid} at path {dir_or_tar_path}, skipping..."
                    )
                    return

                try:
                    with asset_tf.extractfile(f"{uid}/thor_metadata.json") as f:
                        thor_metadata = json.loads(f.read())
                except:
                    pass

        assert asset_ann["uid"] == uid

        if "thor_metadata" in asset_ann:
            warnings.warn(
                f"Warning: THOR metadata already exists for {asset_ann['uid']}, replacing..."
            )
            asset_ann["thor_metadata"] = {}

        if thor_metadata is not None:
            asset_ann["thor_metadata"] = thor_metadata
        else:
            print_or_log(
                f"Warning: No THOR metadata found for {asset_ann['uid']} at path {dir_or_tar_path}."
            )
            asset_ann["thor_metadata"] = {}

        all_annotations[uid] = all_annotations
    finally:
        if pbar is not None:
            pbar.update(1)


def prepare_annotations(save_dir: str, assets_dir: str):
    # For each asset in the assets directory, get the annotations, and save them to a file.

    os.makedirs(save_dir, exist_ok=True)
    all_annotations_save_path = os.path.join(save_dir, "annotations.json.gz")

    if os.path.exists(all_annotations_save_path):
        print(
            f"Annotations already exist at {all_annotations_save_path}. Will load and skip existing."
        )
        all_annotations = compress_json.load(all_annotations_save_path)
    else:
        all_annotations = {}

    # Walk along the assets dir
    paths = sorted(os.scandir(assets_dir), key=lambda x: x.name)
    with tqdm.tqdm(total=len(paths), desc="Compiling annotations") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    update_annotations,
                    dir_or_tar_path=path,
                    all_annotations=all_annotations,
                    pbar=pbar,
                )
                for path in paths
            ]

            wait_for_futures_and_raise_errors(futures)

    # Save the annotations to a file
    compress_json.dump(all_annotations, all_annotations_save_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare annotations for the assets in the assets directory."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="The directory to save the annotations to.",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        required=True,
        help="The directory containing the assets.",
    )
    args = parser.parse_args()

    prepare_annotations(
        save_dir=args.save_dir,
        assets_dir=args.assets_dir,
    )
