import os
import random
import urllib.request
from typing import Dict

import compress_pickle
import numpy as np
from tqdm import tqdm

from objathor.utils.download_utils import download_with_locking
from objathor.utils.gpt_utils import get_embedding, get_embeddings_from_texts
from objathor.utils.synsets import (
    all_synsets,
    synset_definitions,
    synset_lemmas,
    synset_hyponyms,
    synset_hypernyms,
)

OBJATHOR_DATA_DIR = os.path.join(os.path.expanduser("~"), ".objathor_data")

SYNSET_DEFINITION_EMB_FILE = os.path.join(
    OBJATHOR_DATA_DIR, "synset_definition_embeddings_with_lemmas__2024-01-22.pkl.gz"
)


def download_embeddings(
    url: str = "https://pub-daedd7738a984186a00f2ab264d06a07.r2.dev/misc/synset_definition_embeddings_with_lemmas__2024-01-22.pkl.gz",
):
    os.makedirs(OBJATHOR_DATA_DIR, exist_ok=True)
    if not os.path.isfile(SYNSET_DEFINITION_EMB_FILE):
        print(f"Downloading\n{url}\nto\n{SYNSET_DEFINITION_EMB_FILE}")

        download_with_locking(
            url=url,
            save_path=SYNSET_DEFINITION_EMB_FILE,
            lock_path=SYNSET_DEFINITION_EMB_FILE + ".lock",
            desc="Downloading synset definition embeddings",
        )

    assert os.path.isfile(SYNSET_DEFINITION_EMB_FILE)


def get_embeddings(
    fname: str = os.path.join(OBJATHOR_DATA_DIR, "synset_definition_embeddings.pkl.gz"),
) -> Dict[str, np.ndarray]:
    from nltk.corpus import wordnet2022 as wn

    if os.path.isfile(fname):
        data = compress_pickle.load(fname)
    else:
        data = {}

    num_additions = 0
    try:
        synsets = []
        texts = []
        for synset_str in tqdm(all_synsets()):
            if ".n." not in synset_str:
                continue

            if synset_str in data:
                continue

            synset = wn.synset(synset_str)
            lemmas = synset.lemmas()
            name = lemmas[0].name().replace("_", " ")
            other_names = [lemma.name().replace("_", " ") for lemma in lemmas[1:]]

            text = f"{name}: {synset.definition()}."
            if len(other_names) > 0:
                text += f" Also known as a {', '.join(other_names)}."

            synsets.append(synset.name())
            texts.append(text)

        embeddings = get_embeddings_from_texts(texts)
        for synset_str, embedding in zip(synsets, embeddings):
            data[synset_str] = np.array(embedding) / np.linalg.norm(embedding)

        num_additions += len(texts)
    finally:
        if num_additions > 0:
            print("Saving definition embeddings...")
            compress_pickle.dump(data, fname)

    return data


def get_embeddings_single(
    fname: str = SYNSET_DEFINITION_EMB_FILE,
) -> Dict[str, np.ndarray]:
    if not os.path.isfile(fname):
        try:
            download_embeddings()
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            data = get_embeddings()
            for key, value in data.items():
                data[key] = value.astype(np.float32)
            compress_pickle.dump(data, fname)

    return compress_pickle.load(fname)


def local_smoothing(embs: Dict[str, np.ndarray], synset_str: str):
    ref = embs[synset_str]
    from nltk.corpus import wordnet2022 as wn

    comb = [ref]

    hypos = wn.synset(synset_str).hyponyms()
    print("hypos", [syn.name() for syn in hypos])
    if len(hypos) > 0:
        hypos = np.stack([embs[syn.name()] for syn in hypos], axis=1)
        hypo_mean = hypos.sum(axis=1)
        comb.append(0.5 * hypo_mean / np.linalg.norm(hypo_mean))

    hypers = wn.synset(synset_str).hypernyms()
    print("hypers", [syn.name() for syn in hypers])
    if len(hypers) > 0:
        hypers = np.stack([embs[syn.name()] for syn in hypers], axis=1)
        hyper_mean = hypers.sum(axis=1)
        comb.append(0.5 * hyper_mean / np.linalg.norm(hyper_mean))

    comb = np.sum(comb, axis=0)
    comb = comb / np.linalg.norm(comb)

    if len(hypos) > 0:
        print("ref, hypos", ref @ hypos, np.mean(ref @ hypos))

    if len(hypers) > 0:
        print("ref, hypers", ref @ hypers, np.mean(ref @ hypers))

    if len(hypos) > 0 and len(hypers) > 0:
        print("hypos, hypers", hypos.T @ hypers, np.mean(hypos.T @ hypers))

    print("ref, comb", ref @ comb, np.mean(ref @ comb))

    if len(hypos) > 0:
        print("comb, hypos", comb @ hypos, np.mean(comb @ hypos))

    if len(hypers) > 0:
        print("comb, hypers", comb @ hypers, np.mean(comb @ hypers))


def get_lemmas_definition_embeddings(
    fname: str = os.path.join(
        OBJATHOR_DATA_DIR, "synset_lemmas_definitions_embeddings.pkl.gz"
    ),
    max_lemmas: int = 3,
) -> Dict[str, np.ndarray]:
    if os.path.isfile(fname):
        data = compress_pickle.load(fname)
    else:
        data = {}

    def format_lemmas(lemmas):
        lemmas = [f'{lemma.replace("_", " ")}' for lemma in lemmas]

        if len(lemmas) == 0:
            formatted_lemmas = ""
        elif len(lemmas) == 1:
            formatted_lemmas = f"{lemmas[0]}"
        elif len(lemmas) == 2:
            formatted_lemmas = f"{lemmas[0]} or {lemmas[1]}"
        else:
            formatted_lemmas = ", ".join(lemmas[:-1]) + f", or {lemmas[-1]}"

        return formatted_lemmas

    num_additions = 0
    try:
        for synset_str in tqdm(all_synsets()):
            if synset_str in data:
                continue

            lemmas = synset_lemmas([synset_str])[0][:max_lemmas]
            formatted_lemmas = format_lemmas(lemmas)

            lemmas = set(lemmas)

            hyper = (
                set(
                    sum(
                        [
                            synset_lemmas([hyp.name()])[0]
                            for hyp in synset_hypernyms([synset_str])[0]
                        ],
                        [],
                    )
                )
                - lemmas
            )
            hyper = list(hyper)
            random.shuffle(hyper)
            hyper = hyper[:max_lemmas]

            hyper_lemmas = format_lemmas(hyper)

            hyper = set(hyper)

            hypo = (
                set(
                    sum(
                        [
                            synset_lemmas([hyp.name()])[0]
                            for hyp in synset_hyponyms([synset_str])[0]
                        ],
                        [],
                    )
                )
                - lemmas
                - hyper
            )
            hypo = list(hypo)
            random.shuffle(hypo)
            hypo = hypo[:max_lemmas]

            hypo_lemmas = format_lemmas(hypo)

            context = ""
            if len(hypo_lemmas) > 0 and len(hyper_lemmas) > 0:
                context = f", a type of {hyper_lemmas} like {hypo_lemmas}"
            elif len(hyper_lemmas) > 0:
                context = f", a type of {hyper_lemmas}"
            elif len(hypo_lemmas) > 0:
                context = f", like {hypo_lemmas}"

            text = f"{formatted_lemmas}{context}; {synset_definitions([synset_str])[0]}"

            embedding = get_embedding(text)
            data[synset_str] = dict(
                emb=(np.array(embedding) / np.linalg.norm(embedding)).astype(
                    np.float32
                ),
                text=text,
            )

            num_additions += 1
            if num_additions == 1000:
                print(f"Saving definition embeddings with {len(data)} entries...")
                compress_pickle.dump(data, fname)
                num_additions = 0
    finally:
        if num_additions > 0:
            print(f"Saving definition embeddings with {len(data)} entries...")
            compress_pickle.dump(data, fname)

    return data


if __name__ == "__main__":
    data = get_embeddings()
    for key, value in data.items():
        data[key] = value.astype(np.float32)

    compress_pickle.dump(
        data,
        os.path.join(
            OBJATHOR_DATA_DIR,
            "synset_definition_embeddings_with_lemmas__2024-01-22.pkl.gz",
        ),
    )

    # data = get_embeddings()
    # data = get_embeddings_single()
    # local_smoothing(data, "wardrobe.n.01")

    # data = get_lemmas_definition_embeddings()
    print("DONE")
