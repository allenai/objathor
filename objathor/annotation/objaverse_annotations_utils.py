import os.path
from functools import lru_cache

import prior
import torch
from filelock import FileLock


def _lock_for_load():
    path = os.path.expanduser("~/.prior/objaverse-prior.lock")
    return FileLock(os.path.expanduser(path))


@lru_cache(maxsize=1)
def get_objaverse_home_annotations():
    with _lock_for_load():
        return prior.load_dataset(
            "objaverse-plus",
            # revision="ace12898b451c887bb1dd69ede85d32a75a86ef7",  # Human only
            revision="877a5d636a6c437b894d1f8510bc852e49bb1cc0",  # Human + AI
        )["train"].data


@lru_cache(maxsize=1)
def get_objaverse_ref_categories():
    with _lock_for_load():
        annos = prior.load_dataset(
            "objaverse-plus", revision="bce68ddc9f9dfbf1566d61dc4f04ac60e2f2d125"
        )["train"].data
        return {uid: anno["ref_category"] for uid, anno in annos.items()}


@lru_cache(maxsize=1)
def get_objaverse_closest_mapping():
    with _lock_for_load():
        return prior.load_dataset(
            "objaverse-plus",
            revision="877a5d636a6c437b894d1f8510bc852e49bb1cc0",
            which_dataset="closest_mapping",
        )["train"].data


@lru_cache(maxsize=1)
def get_open_clip_vit_l(device: torch.device):
    import open_clip

    clip_model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"

    with _lock_for_load():
        clip_model, _, clip_img_preprocessor = open_clip.create_model_and_transforms(
            model_name=clip_model_name, pretrained=pretrained, device=device
        )
    clip_model.eval()
    return clip_model, clip_img_preprocessor


def compute_clip_vit_l_similarity(img_path0: str, img_path1: str, device: torch.device):
    import torch
    import numpy as np
    from PIL import Image

    clip_model, clip_im_preprocessor = get_open_clip_vit_l(device=device)

    def convert(img_path: str):
        img = np.array(Image.open(img_path).convert("RGBA"), dtype=np.uint8)
        img[img[:, :, 3] == 0] = 255
        img = img[:, :, :3]
        return img

    img0 = convert(img_path0)

    img1 = convert(img_path1)

    with torch.no_grad():
        blender_features = clip_model.encode_image(
            clip_im_preprocessor(Image.fromarray(img0)).unsqueeze(0).to(device)
        )
        thor_features = clip_model.encode_image(
            clip_im_preprocessor(Image.fromarray(img1)).unsqueeze(0).to(device)
        )

        sim = torch.cosine_similarity(blender_features, thor_features)
        return sim.item()
