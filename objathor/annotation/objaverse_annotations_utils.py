from functools import lru_cache

import prior


@lru_cache(maxsize=1)
def get_objaverse_home_annotations():
    return prior.load_dataset(
        "objaverse-plus",
        revision="ace12898b451c887bb1dd69ede85d32a75a86ef7",  # Human only
        # revision="877a5d636a6c437b894d1f8510bc852e49bb1cc0" # Human + AI
    )["train"].data


@lru_cache(maxsize=1)
def get_objaverse_ref_categories():
    annos = prior.load_dataset(
        "objaverse-plus", revision="bce68ddc9f9dfbf1566d61dc4f04ac60e2f2d125"
    )["train"].data
    return {uid: anno["ref_category"] for uid, anno in annos.items()}


@lru_cache(maxsize=1)
def get_objaverse_closest_mapping():
    return prior.load_dataset(
        "objaverse-plus",
        revision="877a5d636a6c437b894d1f8510bc852e49bb1cc0",
        which_dataset="closest_mapping",
    )["train"].data
