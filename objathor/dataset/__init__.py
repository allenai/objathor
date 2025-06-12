import glob
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import attrs
import compress_json
from attrs import define
from filelock import FileLock

from objathor.constants import OBJATHOR_CACHE_PATH
from objathor.utils.download_utils import (
    download_with_progress_bar,
    download_with_locking,
)


@define
class DatasetSaveConfig:
    VERSION: str = attrs.field(default="2023_07_28")
    BASE_PATH: str = attrs.field(default=OBJATHOR_CACHE_PATH)
    BASE_BUCKET_URL: Optional[str] = attrs.field()

    @BASE_BUCKET_URL.default
    def _default_base_bucket_url(self) -> str:
        if self.VERSION == "2025_06_10":
            return "https://pub-ddc5ca49fcee4247b552f4217e910a0f.r2.dev"
        elif self.VERSION == "2024_08_16":
            return "https://pub-2619544d52bd4f35927b08d301d2aba0.r2.dev"
        else:
            return "https://pub-daedd7738a984186a00f2ab264d06a07.r2.dev"

    @property
    def VERSIONED_PATH(self) -> str:
        return os.path.join(self.BASE_PATH, self.VERSION)

    @property
    def VERSIONED_BUCKET_URL(self) -> str:
        if self.VERSION == "2025_06_10":
            return self.BASE_BUCKET_URL
        elif self.VERSION == "2024_08_16":
            return self.BASE_BUCKET_URL
        else:
            return f"{self.BASE_BUCKET_URL}/{self.VERSION}"

    @property
    def ANNOTATIONS_LOCK(self) -> str:
        return os.path.join(self.VERSIONED_PATH, "annotations.lock")

    @property
    def OBJECTS_LOCK(self) -> str:
        return os.path.join(self.VERSIONED_PATH, "objects.lock")

    @property
    def HD_LOCK(self) -> str:
        return os.path.join(self.VERSIONED_PATH, "hd.lock")


DEFAULT_DSC = DatasetSaveConfig()


def load_annotations_path(dsc: DatasetSaveConfig = DEFAULT_DSC) -> str:
    annotations_file_name = "annotations.json.gz"
    annotations_path = os.path.join(dsc.VERSIONED_PATH, annotations_file_name)

    download_with_locking(
        url=f"{dsc.VERSIONED_BUCKET_URL}/{annotations_file_name}",
        save_path=annotations_path,
        lock_path=dsc.ANNOTATIONS_LOCK,
        desc="Downloading annotations.",
    )
    return annotations_path


def load_annotations(dsc: DatasetSaveConfig = DEFAULT_DSC) -> Dict[str, Any]:
    return compress_json.load(load_annotations_path(dsc))


def load_assets_path(
    dsc: DatasetSaveConfig = DEFAULT_DSC,
) -> str:
    os.makedirs(dsc.VERSIONED_PATH, exist_ok=True)
    asset_save_path = os.path.join(dsc.VERSIONED_PATH, "assets")

    with FileLock(dsc.OBJECTS_LOCK):
        if not os.path.exists(asset_save_path):
            tar_path = asset_save_path + ".tar"
            download_with_progress_bar(
                url=f"{dsc.VERSIONED_BUCKET_URL}/assets.tar",
                save_path=tar_path,
                desc="Downloading assets.",
            )

            tmp_untar_dir = os.path.join(dsc.VERSIONED_PATH, "tmp_untar_dir")
            os.makedirs(tmp_untar_dir, exist_ok=True)

            print(f"Untar-in {tar_path}, this may take a while...")
            os.system(f"tar -xf {tar_path} -C {tmp_untar_dir}")

            paths = glob.glob(os.path.join(tmp_untar_dir, "*"))
            assert len(paths) == 1

            print(f"Moving {paths[0]} to {asset_save_path}")

            os.rename(paths[0], asset_save_path)

            print(f"Removing {tmp_untar_dir} and {tar_path}...")
            os.rmdir(tmp_untar_dir)
            os.remove(tar_path)

    return asset_save_path


def load_features_dir(
    dsc: DatasetSaveConfig = DEFAULT_DSC,
) -> str:
    features_save_dir = os.path.join(dsc.VERSIONED_PATH, "features")
    os.makedirs(features_save_dir, exist_ok=True)

    with FileLock(dsc.OBJECTS_LOCK):
        if (not os.path.exists(features_save_dir)) or len(
            glob.glob(os.path.join(features_save_dir, "**", "*.pkl"), recursive=True)
        ) == 0:
            tar_path = features_save_dir + ".tar"
            try:
                download_with_progress_bar(
                    url=f"{dsc.VERSIONED_BUCKET_URL}/features.tar",
                    save_path=tar_path,
                    desc="Downloading features.",
                )
            except ValueError:
                # Fallback to assuming BASE_BUCKET_URL is a link to the bucket root
                # with a flat structure.
                download_with_progress_bar(
                    url=f"{dsc.BASE_BUCKET_URL}/features.tar",
                    save_path=tar_path,
                    desc="Downloading features.",
                )

            _td = TemporaryDirectory()
            with _td as tmp_untar_dir:
                print(f"Untar-in {tar_path}, this may take a while...")
                os.system(f"tar -xf {tar_path} -C {tmp_untar_dir}")

                paths = glob.glob(
                    os.path.join(tmp_untar_dir, "**", "*.pkl"), recursive=True
                )
                assert len(paths) > 1

                for p in paths:
                    print(f"Moving {p} to {features_save_dir}")
                    os.rename(p, os.path.join(features_save_dir, os.path.basename(p)))

                print(f"Removing {tmp_untar_dir} and {tar_path}...")
                os.remove(tar_path)

    return features_save_dir


def load_holodeck_base(
    dsc: DatasetSaveConfig = DEFAULT_DSC,
) -> str:
    base_dir = os.path.join(dsc.BASE_PATH, "holodeck")
    os.makedirs(base_dir, exist_ok=True)

    save_dir = os.path.join(base_dir, dsc.VERSION)
    with FileLock(dsc.HD_LOCK):
        if os.path.exists(save_dir):
            print(f"Holodeck base already exists at {save_dir}")
        else:
            tar_path = save_dir + ".tar"
            download_with_progress_bar(
                url=f"{dsc.BASE_BUCKET_URL}/holodeck/{dsc.VERSION}.tar",
                save_path=tar_path,
                desc="Downloading holodeck base files.",
            )

            _td = TemporaryDirectory()
            with _td as tmp_untar_dir:
                print(f"Untar-in {tar_path}, this may take a while...")
                os.system(f"tar -xf {tar_path} -C {tmp_untar_dir}")

                paths = glob.glob(os.path.join(tmp_untar_dir, "*"))
                assert len(paths) == 1

                print(f"Moving {paths[0]} to {save_dir}")
                os.rename(paths[0], save_dir)

                print(f"Removing {tar_path}...")
                os.remove(tar_path)

    return save_dir
