from typing import Dict
import os

import compress_pickle
import numpy as np
from tqdm import tqdm

from utils.gpt_utils import get_embedding
from utils.synsets import all_synsets, synset_definitions


def get_embeddings(
    fname: str = "data/synset_definition_embeddings.pkl.gz",
) -> Dict[str, np.ndarray]:
    if os.path.isfile(fname):
        data = compress_pickle.load(fname)
    else:
        data = {}

    num_additions = 0
    try:
        for synset_str in tqdm(all_synsets()):
            if synset_str in data:
                continue
            definition = synset_definitions([synset_str])[0]
            embedding = get_embedding(definition)
            data[synset_str] = np.array(embedding) / np.linalg.norm(embedding)
            num_additions += 1
    finally:
        if num_additions > 0:
            print("Saving definition embeddings...")
            compress_pickle.dump(data, fname)

    return compress_pickle.load(fname)


if __name__ == "__main__":
    get_embeddings()
    print("DONE")
