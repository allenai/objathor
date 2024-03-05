import os
import subprocess
from typing import List, Optional, Sequence


def import_install_nltk_commit(
    commit_id="582e6e35f0e6c984b44ec49dcb8846d9c011d0a8", install_if_import_fails=True
):
    try:
        import nltk
    except ImportError as e:
        # print(f"Coululd not import `nltk` please install it by running `pip install git+https://github.com/nltk/nltk@{commit_id}`.")
        raise ValueError(
            f"\nCoululd not import `nltk` please install it by running `pip install git+https://github.com/nltk/nltk@{commit_id}`."
        )
        # if install_if_import_fails:

        # does not work when subprocess.
        # errors: fatal: fetch-pack: invalid index-pack output; error: subprocess-exited-with-error
        # command = (
        #     f"pip"
        #     f" install"
        #     f" git+https://github.com/nltk/nltk@{commit_id}"
        # )
        # try:
        #     print(f"Installing nltk commit_id={commit_id}, running: `{command}`")
        #     subprocess.check_call(command, shell=True)
        # except Exception as e:
        #     result_code = e.returncode
        #     print(f"`pip install` process error: {e.output}")
        #     out = e.output
        # else:
        # print(f"Coululd not import `nltk` please install it by running `pip install git+https://github.com/nltk/nltk@{commit_id}`.")


try:
    from nltk.corpus import wordnet2022 as wn
except ImportError:
    # import nltk
    import_install_nltk_commit()

    nltk.download("wordnet2022")
    nltk.download("punkt")
    nltk.download("brown")
    nltk.download("averaged_perceptron_tagger")

    from nltk.corpus import wordnet2022 as wn


DEFAULT_TOP_SYNSET_STR = "entity.n.01"
# DEFAULT_TOP_SYNSET_STR = "physical_entity.n.01"
# DEFAULT_TOP_SYNSET_STR = "object.n.01"


def _gather_synsets(current_synset_str: str) -> List[str]:
    res = []

    for hyponym in wn.synset(current_synset_str).hyponyms():
        res.extend(_gather_synsets(hyponym.name()))

    res.append(current_synset_str)

    return res


_ALL_SYNSETS = {}


def all_synsets(top_synset_str: Optional[str] = None) -> List[str]:
    top_synset_str = top_synset_str or DEFAULT_TOP_SYNSET_STR

    global _ALL_SYNSETS
    if _ALL_SYNSETS.get(top_synset_str, None) is None:
        _ALL_SYNSETS[top_synset_str] = sorted(
            list(set(_gather_synsets(top_synset_str)))
        )
    return _ALL_SYNSETS[top_synset_str]


def synset_definitions(synset_strs: Sequence[str]):
    return [wn.synset(synset_str).definition() for synset_str in synset_strs]


def synset_lemmas(synset_strs: Sequence[str]):
    return [wn.synset(synset_str).lemma_names() for synset_str in synset_strs]


def synset_hyponyms(synset_strs: Sequence[str]):
    return [wn.synset(synset_str).hyponyms() for synset_str in synset_strs]


def synset_hypernyms(synset_strs: Sequence[str]):
    return [wn.synset(synset_str).hypernyms() for synset_str in synset_strs]
