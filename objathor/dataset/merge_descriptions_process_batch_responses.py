import json
import os.path
from collections import Counter

from tqdm import tqdm

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
    word_count = Counter()
    with open(raw_responses_path) as f:
        for line in tqdm(f, "reading"):
            for custom_id, merged in json.loads(line).items():
                idx = int(custom_id.split("_")[1])
                assert idx not in processed_idxs
                assert idx < len(filtered_ids)
                processed_idxs.add(idx)

                for word in merged.split():
                    word_frequency[word] += 1
                word_count[len(merged.split())] += 1

                if len(merged.split()) < 3:
                    print("SHORT", idx, filtered_ids[idx], merged)
                    short_descriptions += 1
                elif len(merged.split()) > 6:
                    print("LONG", idx, filtered_ids[idx], merged)
                    long_descriptions += 1
                if not 3 <= len(merged.split()) <= 6:
                    print([value for key, value in SharedJSONDatabase()[filtered_ids[idx]].items() if "description" in key])

                full_data.append(dict(uuid=filtered_ids[idx], description=merged))

    print(f"Word frequencies {word_frequency.most_common()}")

    mean_length = sum(key * val for key, val in word_count.items()) / word_count.total()
    print(f"Word counts {word_count.most_common()}, mean {mean_length:.2f} words per descriptor")
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
                "/weka/prior/jordis/merge_descriptions_raw/raw_merge_descriptions_batch_ids.jsonl"
            )
        )

        with open(
            os.path.expanduser("/weka/prior/jordis/merge_descriptions_aggregate.json"),
            "w",
        ) as f:
            json.dump(aggregate, f)

        print("DONE")

    main()
