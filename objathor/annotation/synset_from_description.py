from typing import Sequence, Optional, Dict, List
import os

import compress_pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

from objathor.utils.gpt_utils import get_embedding
from objathor.annotation.embed_synset_definitions import (
    get_embeddings_single as get_synset_embeddings,
)

OUTPUT_DIR = "/tmp/objathor_description_embeddings"
NUM_NEIGHS = 5


_SYNSET_EMBEDDINGS: Optional[Dict[str, np.ndarray]] = None


def synset_embeddings():
    global _SYNSET_EMBEDDINGS
    if _SYNSET_EMBEDDINGS is None:
        _SYNSET_EMBEDDINGS = get_synset_embeddings()
        print(f"Loaded {len(_SYNSET_EMBEDDINGS)} embeddings")
    return _SYNSET_EMBEDDINGS


_KEYS: Optional[List[str]] = None
_NN: Optional[NearestNeighbors] = None


def keys():
    global _KEYS
    if _KEYS is None:
        _KEYS = sorted(list(synset_embeddings().keys()))
    return _KEYS


def get_nn():
    global _NN
    if _NN is None:
        values = np.stack([synset_embeddings()[key] for key in keys()])
        print(
            f"NN data shape {values.shape} ({len(keys())} keys), {NUM_NEIGHS} neighbors"
        )
        _NN = NearestNeighbors(n_neighbors=NUM_NEIGHS).fit(values)
    return _NN


def normalize_embedding(emb: Sequence[float]):
    return (np.array(emb) / np.linalg.norm(emb)).astype(np.float16)


def save_embedding(emb: Sequence[float], uid: str, dir: str = OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compress_pickle.dump(
        normalize_embedding(emb),
        os.path.join(OUTPUT_DIR, f"description_emb_{uid}.pkl.gz"),
    )


def embed_description(annotation):
    description = annotation["description"]
    return get_embedding(description)


def nearest_synsets_from_annotation(annotation, save_to_dir: Optional[str] = None):
    desc_emb = embed_description(annotation)
    if save_to_dir is not None:
        try:
            save_embedding(desc_emb, uid=annotation["uid"], dir=save_to_dir)
        except Exception as e:
            print("ERROR saving description embedding", e)
    inds = get_nn().kneighbors(
        normalize_embedding(desc_emb).reshape(1, -1), return_distance=False
    )[0]
    return [keys()[ind] for ind in inds]
