import json
import os.path
from collections import Counter

from tqdm import tqdm

from objathor.dataset.short_descriptions import DescriptionOutput
from objathor.dataset.merge_descriptions import SharedJSONDatabase, get_split_to_whitelisted_assets


def raw_to_aggregate(raw_responses_path):
    filtered_ids = sorted(
        set(
            sum(
                [
                    split_items
                    for split, split_items in get_split_to_whitelisted_assets().items()
                ],
                [],
            )
        )
        & set(SharedJSONDatabase().keys())
    )

    full_data = []

    processed_idxs = set()
    short_descriptions = 0
    long_descriptions = 0
    word_frequency = Counter()
    word_count = dict(
        one_word=Counter(),
        two_words=Counter(),
        three_words=Counter(),
        four_words=Counter(),
        five_words=Counter(),
    )
    expected_length=dict(
        one_word=1,
        two_words=2,
        three_words=3,
        four_words=4,
        five_words=5,
    )
    with open(raw_responses_path) as f:
        for line in tqdm(f, "reading"):
            for custom_id, merged in json.loads(line).items():
                idx = int(custom_id.split("_")[1])
                assert idx not in processed_idxs
                assert idx < len(filtered_ids)
                processed_idxs.add(idx)

                try:
                    merged = DescriptionOutput.model_validate_json(merged).model_dump()
                except ValidationError as e:
                    print(e)
                    continue

                for wc, description in merged.items():
                    for word in description.split():
                        word_frequency[word] += 1
                    word_count[wc][len(description.split())] += 1

                    if len(description.split()) < expected_length[wc]:
                        print("SHORT", wc, idx, filtered_ids[idx], description)
                        short_descriptions += 1
                    elif len(description.split()) > expected_length[wc]:
                        print("LONG", wc, idx, filtered_ids[idx], description)
                        long_descriptions += 1

                full_data.append(dict(uuid=filtered_ids[idx], description=merged))

    print(f"Word frequencies {word_frequency.most_common()}")

    for wc in word_count:
        mean_length = sum(key * val for key, val in word_count[wc].items()) / word_count[wc].total()
        print(f"Word counts {wc} {word_count[wc].most_common()}, mean {mean_length:.2f} words per descriptor")
    print(
        f"{len(full_data)} out of {len(filtered_ids)} returned with {short_descriptions=} and {long_descriptions=}"
    )

    missed_idxs = set(range(len(filtered_ids))) - processed_idxs
    missed_uuids = [filtered_ids[idx] for idx in missed_idxs]
    for missed_uuid in missed_uuids:
        print(f"Missed {missed_uuid}")
        print([value for key, value in SharedJSONDatabase()[missed_uuid].items() if "description" in key])

    return full_data


if __name__ == "__main__":

    def main():
        aggregate = raw_to_aggregate(
            os.path.expanduser(
                "/weka/prior/jordis/short_descriptions/raw_output/raw_short_descriptions_batch_ids.jsonl"
            )
        )

        with open(
            os.path.expanduser("/weka/prior/jordis/short_descriptions/short_descriptions_aggregate.json"),
            "w",
        ) as f:
            json.dump(aggregate, f)

        print("DONE")

    main()
