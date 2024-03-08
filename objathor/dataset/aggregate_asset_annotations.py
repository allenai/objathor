import os
import warnings
from argparse import ArgumentParser

import compress_json
import tqdm


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


def prepare_annotations(save_dir: str, assets_dir: str):
    # For each asset in the assets directory, get the annotations, and save them to a file.

    os.makedirs(save_dir, exist_ok=True)
    all_annotations_save_path = os.path.join(save_dir, "annotations.json.gz")

    if os.path.exists(all_annotations_save_path):
        print(f"Annotations already exist at {all_annotations_save_path}. Skipping...")
        return

    # Walk along the assets dir
    all_annotations = {}
    for dir in tqdm.tqdm(os.scandir(assets_dir)):
        if os.path.isdir(dir):
            annotations_path = os.path.join(dir, "annotations.json.gz")
            thor_metadata_path = os.path.join(dir, "thor_metadata.json")

            if os.path.exists(annotations_path):
                annotations = compress_json.load(annotations_path)

                assert annotations["uid"] == os.path.basename(dir)

                if "thor_metadata" in annotations:
                    warnings.warn(
                        f"Warning: THOR metadata already exists for {annotations['uid']}, replacing..."
                    )
                    annotations["thor_metadata"] = {}

                if os.path.exists(thor_metadata_path):
                    annotations["thor_metadata"] = compress_json.load(
                        thor_metadata_path
                    )
                else:
                    print(
                        f"Warning: No THOR metadata found for {annotations['uid']} at path {thor_metadata_path}."
                    )
                    annotations["thor_metadata"] = {}

                all_annotations[annotations["uid"]] = annotations

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
