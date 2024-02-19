import glob
import os
from typing import Any, Dict

import compress_json
import requests
from attrs import define
from filelock import FileLock
from tqdm import tqdm


def download_with_progress_bar(url: str, save_path: str, desc: str = ""):
    with open(save_path, "wb") as f:
        print(f"Downloading {save_path}")
        response = requests.get(url, stream=True)
        total_length = response.headers.get("content-length")

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            with tqdm(total=total_length, unit="B", unit_scale=True, desc=desc) as pbar:
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    pbar.update(len(data))


@define
class DatasetSaveConfig:
    VERSION = "2023-07-28"
    BASE_PATH = os.path.join(os.path.expanduser("~"), ".objathor-assets")

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


def download_with_locking(url: str, save_path: str, lock_path: str, desc: str = ""):
    if not os.path.exists(save_path):
        with FileLock(lock_path):
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                download_with_progress_bar(url=url, save_path=save_path, desc=desc)

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

