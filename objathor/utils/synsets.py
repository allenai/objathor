from typing import List, Optional, Sequence

try:
    from nltk.corpus import wordnet2022 as wn
except:
    import nltk

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
