import glob
import os
from argparse import ArgumentParser
from typing import Dict, Callable, Union, Sequence

import PIL.Image as Image
import compress_json
import compress_pickle
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

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


class ObjectDataset(Dataset):
    def __init__(
        self,
        asset_dir: str,
        annotations: Dict[str, ObjectAnnotation],
        image_preprocessor: Callable,
        img_angles: Sequence[float] = (0.0, 45.0, 315.0),
    ):
        self.annotations = annotations
        self.asset_dir = asset_dir
        self.image_preprocessor = image_preprocessor
        self.img_angles = img_angles

        self.uids = sorted(self.annotations.keys())

        if self.image_preprocessor is not None:
            assert len(img_angles) > 0, "At least one angle must be provided."

            render_paths = glob.glob(
                os.path.join(
                    self.asset_dir, f"*/thor_renders/0_1_0_{img_angles[0]:.1f}.*"
                )
            )

            for rp in render_paths:
                for angle in img_angles[1:]:
                    assert os.path.exists(
                        rp.replace(f"0_1_0_{img_angles[0]:.1f}", f"0_1_0_{angle:.1f}")
                    ), f"Missing render for {os.path.dirname(rp)} at angle {angle}."

            render_uids = set(
                os.path.basename(os.path.dirname(os.path.dirname(p)))
                for p in render_paths
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
            # angle = float(np.rad2deg(ann["pose_z_rot_angle"])) # Used for blender renders

            for angle in self.img_angles:
                img_path = os.path.join(
                    self.asset_dir, uid, "thor_renders", f"0_1_0_{angle:.1f}.jpg"
                )
                if not os.path.exists(img_path):
                    img_path = img_path.replace(".png", ".jpg")

                item[f"img_{angle:.1f}"] = self.image_preprocessor(Image.open(img_path))

        return item


def generate_features(
    base_dir: str,
    annotations_path: str,
    device: str,
    batch_size: int,
    num_workers: int,
):
    # CLIP
    device = torch.device(device)
    clip_model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"
    clip_model, _, clip_img_preprocessor = open_clip.create_model_and_transforms(
        model_name=clip_model_name, pretrained=pretrained, device=device
    )
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    # Sentence BERT
    sbert_model = SentenceTransformer("all-mpnet-base-v2").to(device)

    dataset = ObjectDataset(
        annotations=compress_json.load(annotations_path),
        asset_dir=assets_dir,
        image_preprocessor=clip_img_preprocessor,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    uids = []
    clip_img_features = []
    clip_text_features = []
    sbert_text_features = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for batch in dataloader:
                uids.extend(batch["uid"])

                clip_img_features_per_angle = []
                for angle in dataset.img_angles:
                    clip_img_features_per_angle.append(
                        clip_model.encode_image(batch[f"img_{angle:.1f}"].to(device))
                    )

                clip_img_features.append(
                    torch.stack(clip_img_features_per_angle, dim=0).cpu()
                )

                clip_text_features.append(
                    clip_model.encode_text(
                        clip_tokenizer(batch["text"]).to(device)
                    ).cpu()
                )
                sbert_text_features.append(
                    sbert_model.encode(
                        batch["text"], convert_to_tensor=True, show_progress_bar=False
                    ).cpu()
                )

                pbar.update(len(batch["uid"]))

    clip_img_features = torch.cat(clip_img_features, dim=0).numpy()
    clip_text_features = torch.cat(clip_text_features, dim=0).numpy()
    sbert_text_features = torch.cat(sbert_text_features, dim=0).numpy()

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


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to process annotated assets for use in Holodeck."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.abspath(
            os.path.join(ABS_PATH_OF_OBJATHOR, "output", "dataset")
        ),
        help="Base directory for datasets.",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        help="Assets directory, will default to <base_dir>/assets if not specified.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Torch device to be used by the models.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for DataLoader."
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoader."
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="",
        help="Path to the annotations file for all objects. See the aggregate_asset_annotations.py script."
        "Defaults to <base_dir>/annotations.json.gz if not specified.",
    )
    args = parser.parse_args()

    # Setting up save / loading paths
    assets_dir = (
        args.assets_dir if args.assets_dir else os.path.join(args.base_dir, "assets")
    )
    annotations_path = (
        args.annotations_path
        if args.annotations_path != ""
        else os.path.join(args.base_dir, "annotations.json.gz")
    )

    generate_features(
        base_dir=args.base_dir,
        annotations_path=annotations_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
