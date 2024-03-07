import os
import subprocess
from argparse import ArgumentParser


def rlcone_upload(
    path_to_upload: str, bucket_path: str, less_safe_more_fast: bool = False
):
    rclone_command = [
        "rclone",
        "copy",
        "-P",
    ]
    if less_safe_more_fast:
        rclone_command.extend(
            [
                "--s3-chunk-size",
                "50M",
                "--transfers",
                "300",
                "--s3-upload-concurrency",
                "300",
                "--s3-chunk-size",
                "50M",
                "--ignore-checksum",
                "--s3-disable-checksum",
            ]
        )
    rclone_command.extend([path_to_upload, bucket_path])

    subprocess.run(rclone_command, check=True)


def upload_dataset(dataset_dir: str, bucket_path: str):
    # Upload annotations
    paths = [
        os.path.join(dataset_dir, name)
        for name in [
            "annotations.json.gz",
            "assets.tar",
            "clip_features.pkl",
            "sbert_features.pkl",
        ]
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"Uploading {path} to {bucket_path}...")
            if os.path.getsize(path) > 100_000_000:  # 100MB
                print(
                    f"File size of {path} is greater than 100MB. Using less safe but faster upload method"
                )
                less_safe_more_fast = True
            else:
                less_safe_more_fast = False

            rlcone_upload(
                path_to_upload=path,
                bucket_path=bucket_path,
                less_safe_more_fast=less_safe_more_fast,
            )
        else:
            print(f"{path} does not exist. Skipping upload...")


if __name__ == "__main__":
    parser = ArgumentParser(description="Script to upload dataset to a bucket.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory, files to upload should be found under this directory.",
    )
    parser.add_argument(
        "--bucket_path",
        type=str,
        required=True,
        help="Bucket to upload the data to.",
    )
    args = parser.parse_args()
    assert (
        os.path.dirname(args.base_dir) == args.bucket_path.split("/")[-1]
    ), f"The base_dir should be the parent directory of the bucket_path. base_dir={args.base_dir}, bucket_path={args.bucket_path}"

    upload_dataset(dataset_dir=args.base_dir, bucket_path=args.bucket_path)
