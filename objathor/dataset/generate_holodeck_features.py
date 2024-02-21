import glob
import os
from typing import Dict, Callable, Union

import PIL.Image as Image
import compress_json
import compress_pickle
import numpy as np
import torch
import tqdm

from objathor.annotation.annotation_utils import ObjectAnnotation
from objathor.constants import ABS_PATH_OF_OBJATHOR

try:
    import open_clip
except ImportError:
    raise ImportError(
        f"open_clip is not installed, make sure to either run 'pip install open_clip_torch' to install it."
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        f"sentence_transformers is not installed, make sure to run 'pip install sentence_transformers' to install it."
    )

from torch.utils.data import Dataset, DataLoader


class ObjectDataset(Dataset):
    def __init__(
        self,
        asset_dir: str,
        annotations: Dict[str, ObjectAnnotation],
        image_preprocessor: Callable,
    ):
        self.annotations = annotations
        self.asset_dir = asset_dir
        self.image_preprocessor = image_preprocessor

        self.uids = sorted(self.annotations.keys())

        render_paths = glob.glob(
            os.path.join(self.asset_dir, f"*/blender_renders/render_0.0.png")
        )
        render_uids = set(
            os.path.basename(os.path.dirname(os.path.dirname(p))) for p in render_paths
        )

        assert (
            set(self.annotations.keys()) - render_uids
        ), f"Some objects with annotations are missing renders: {set(self.annotations.keys()) - render_uids}."

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, torch.Tensor]]:
        uid = self.uids[idx]
        ann = self.annotations[uid]

        item = {
            "idx": idx,
            "uid": uid,
            "text": ann["description"],
        }

        if self.image_preprocessor is not None:
            angle = float(np.rad2deg(ann["pose_z_rot_angle"]))
            img_path = os.path.join(
                self.asset_dir, uid, "blender_renders", f"render_{angle:.1f}.png"
            )
            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")

            item["img"] = self.image_preprocessor(Image.open(img_path))

        return item


if __name__ == "__main__":
    # Setting up save / loading paths
    base_dir = os.path.abspath(
        os.path.join(ABS_PATH_OF_OBJATHOR, "..", "output", "dataset-test")
    )
    assets_dir = os.path.join(base_dir, "assets")

    # CLIP
    clip_model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"
    clip_model, _, clip_img_preprocessor = open_clip.create_model_and_transforms(
        model_name=clip_model_name, pretrained=pretrained
    )
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    # Sentence BERT
    sbert_model = SentenceTransformer("all-mpnet-base-v2")

    dataset = ObjectDataset(
        annotations=compress_json.load(os.path.join(base_dir, "annotations.json.gz")),
        asset_dir=assets_dir,
        image_preprocessor=clip_img_preprocessor,
    )
    idx_to_uid = dataset.uids

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    uids = []
    clip_img_features = []
    clip_text_features = []
    sbert_text_features = []
    with torch.no_grad():
        with tqdm.tqdm(dataloader, total=len(dataset)) as pbar:
            for batch in pbar:
                uids.extend(batch["uid"])
                clip_img_features.append(clip_model.encode_image(batch["img"]))
                clip_text_features.append(
                    clip_model.encode_text(clip_tokenizer(batch["text"]))
                )
                sbert_text_features.append(
                    sbert_model.encode(
                        batch["text"], convert_to_tensor=True, show_progress_bar=False
                    )
                )

                pbar.update(len(uids))

    clip_img_features = torch.cat(clip_img_features, dim=0).cpu().numpy()
    clip_text_features = torch.cat(clip_text_features, dim=0).cpu().numpy()
    sbert_text_features = torch.cat(sbert_text_features, dim=0).cpu().numpy()

    compress_pickle.dump(
        {
            "uids": uids,
            "img_features": clip_img_features,
            "text_features": clip_text_features,
        },
        os.path.join(base_dir, "clip_features.pkl"),
    )

    compress_pickle.dump(
        {
            "uids": uids,
            "text_features": sbert_text_features,
        },
        os.path.join(base_dir, "sbert_features.pkl"),
    )
