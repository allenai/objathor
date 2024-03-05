import os
import traceback
from typing import Sequence, Optional, Dict, List, Tuple

import compress_pickle
import numpy as np

from objathor.utils.synsets import import_install_nltk_commit

# import nltk
import_install_nltk_commit()
from nltk.corpus import wordnet2022 as wn
from sklearn.neighbors import NearestNeighbors

from objathor.annotation.embed_synset_definitions import (
    get_embeddings_single as get_synset_embeddings,
)
from objathor.utils.gpt_utils import get_embedding

DESCRIPTION_EMBEDDING_OUTPUT_DIR = "/tmp/objathor_description_embeddings"
NUM_NEIGHS = 5


_SYNSET_EMBEDDINGS: Optional[Dict[str, np.ndarray]] = None


PICK_SINGLE_SYNSET_TEMPLATE = """\
Below are a list of synsets from WordNet2022 along with their definitions, lemmas, hypernyms, and hyponyms.\
 Pick exactly one synset that best describes the object in the image and respond with that synset's ID. Include\
 no other text in your response.
"""

PICK_SINGLE_SYNSET_USING_OBJECT_INFO_TEMPLATE = """\
I have an object with description:

"{description}"

This object is approximately {scale:0.3g} meters tall.

Below are a list of synsets from WordNet2022 along with their definitions, lemmas, hypernyms, and hyponyms.\
 Pick exactly one synset that best describes the above object's type and respond with that synset's ID. Include\
 no other text in your response.
"""


SYNSET_DESCRIPTION_TEMPLATE = """\
SYNSET ID: {synset_id}
DEFINITION: {definition}
LEMMAS: {lemmas}
HYPERNYMS: {hypernyms}
HYPONYMS: {hyponyms}"""


def synset_to_summary_str(synset: str) -> str:
    from nltk.corpus import wordnet2022 as wn

    s = wn.synset(synset)

    return SYNSET_DESCRIPTION_TEMPLATE.format(
        synset_id=s.name(),
        definition=s.definition(),
        hypernyms=", ".join([h.name() for h in s.hypernyms()][:5]),
        hyponyms=", ".join([h.name() for h in s.hyponyms()][:5]),
        lemmas=", ".join([l.name() for l in s.lemmas()][:5]),
    )


def prompt_for_best_synset(synsets: Sequence[str]) -> str:
    return (
        PICK_SINGLE_SYNSET_TEMPLATE
        + "\n"
        + "\n\n".join([synset_to_summary_str(s) for s in synsets])
    )


def synset_embeddings():
    global _SYNSET_EMBEDDINGS
    if _SYNSET_EMBEDDINGS is None:
        _SYNSET_EMBEDDINGS = get_synset_embeddings()
        print(f"Loaded {len(_SYNSET_EMBEDDINGS)} embeddings")
    return _SYNSET_EMBEDDINGS


_KEYS: Optional[List[str]] = None
_NN: Optional[NearestNeighbors] = None


def all_embedded_synset() -> List[str]:
    global _KEYS
    if _KEYS is None:
        _KEYS = sorted(list(synset_embeddings().keys()))
    return _KEYS


def nearest_neighbor_synsets() -> NearestNeighbors:
    global _NN
    if _NN is None:
        values = np.stack([synset_embeddings()[key] for key in all_embedded_synset()])
        print(
            f"NN data shape {values.shape} ({len(all_embedded_synset())} keys), {NUM_NEIGHS} neighbors"
        )
        _NN = NearestNeighbors(n_neighbors=NUM_NEIGHS).fit(values)
    return _NN


def normalize_embedding(emb: Sequence[float]):
    return (np.array(emb) / np.linalg.norm(emb)).astype(np.float16)


def save_embedding(
    emb: Sequence[float], uid: str, dir: str = DESCRIPTION_EMBEDDING_OUTPUT_DIR
):
    os.makedirs(dir, exist_ok=True)
    compress_pickle.dump(
        normalize_embedding(emb),
        os.path.join(dir, f"description_emb_{uid}.pkl.gz"),
    )


def embed_description(annotation):
    description = annotation["description"]
    return get_embedding(description)


def nearest_synsets_from_annotation(
    annotation, save_to_dir: Optional[str] = None, n_neighbors: int = NUM_NEIGHS
) -> Tuple[List[str], List[float]]:
    text = annotation["description"]

    if annotation.get("category", "").strip() != "":
        text = f"A {annotation['category']}. {text}"

    desc_emb = get_embedding(text)
    if save_to_dir is not None:
        try:
            save_embedding(emb=desc_emb, uid=annotation["uid"], dir=save_to_dir)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            print(
                f"ERROR saving description embedding {e}. Traceback:\n{traceback.format_exc()}"
            )

    dists, inds = nearest_neighbor_synsets().kneighbors(
        normalize_embedding(desc_emb).reshape(1, -1),
        return_distance=True,
        n_neighbors=n_neighbors,
    )
    dists = dists[0]
    inds = inds[0]

    return [all_embedded_synset()[ind] for ind in inds], dists


def synsets_from_text(text: str) -> List[str]:
    is_noun = lambda pos: pos[:2] == "NN"
    tokenized = nltk.word_tokenize(text)
    nouns = [
        word.lower().strip() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)
    ]

    possible_lemmas = ["_".join(text.lower().strip().split(" ")), *nouns]

    synsets = []
    for pl in possible_lemmas:
        for s in wn.synsets(pl):
            s_str = s.name()
            if ".n." in s_str and s_str:
                synsets.append(s.name())

    return list(set(synsets))
