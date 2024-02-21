import glob
import os
from typing import Any, Dict

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
    VERSION = "2023-07-28"
    BASE_PATH = OBJATHOR_CACHE_PATH

    BASE_BUCKET_URL = "https://pub-daedd7738a984186a00f2ab264d06a07.r2.dev"

    @property
    def VERSIONED_PATH(self) -> str:
        return os.path.join(self.BASE_PATH, self.VERSION)

    @property
    def BUCKET_URL(self) -> str:
        return f"{self.BASE_BUCKET_URL}/{self.VERSION}"

    @property
    def ANNOTATIONS_LOCK(self) -> str:
        return os.path.join(self.VERSIONED_PATH, "annotations.lock")

    @property
    def OBJECTS_LOCK(self) -> str:
        return os.path.join(self.VERSIONED_PATH, "objects.lock")


DEFAULT_DSC = DatasetSaveConfig()


def load_annotations_path(dsc: DatasetSaveConfig = DEFAULT_DSC) -> str:
    annotations_file_name = "annotations.json.gz"
    annotations_path = os.path.join(dsc.VERSIONED_PATH, annotations_file_name)

    download_with_locking(
        url=f"{dsc.BUCKET_URL}/{annotations_file_name}",
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
                url=f"{dsc.BUCKET_URL}/assets.tar",
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

            print(f"Assets saved to {asset_save_path}")

    return asset_save_path
