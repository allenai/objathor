import json
import os.path
from io import BytesIO
from typing import List, Tuple, Sequence, Any, Optional, Dict, TypedDict

import requests

from objathor.annotation.synset_from_description import (
    nearest_synsets_from_annotation,
    DESCRIPTION_EMBEDDING_OUTPUT_DIR,
    prompt_for_best_synset,
    NUM_NEIGHS,
    synsets_from_text,
)
from objathor.utils.gpt_utils import get_answer
from objathor.utils.queries import Text, Image, ComposedMessage

DEFAULT_THUMBNAIL_SOURCE_URL = "https://objaverse-im.s3.us-west-2.amazonaws.com"
DEFAULT_VIEW_INDICES = ("0", "3", "6", "9")

DEFAULT_QUESTION = """Please annotate this 3D asset, corresponding to an object that can be found in a home, with the following values (output valid JSON, without additional comments):
"annotations": {
    "description": a description of the object (don't use the term "3D asset" or similar here),
    "synset": the synset of the object that is most closely related. Try to be as specific as possible. This could be "cat.n.01", "glass.n.03", "bank.n.02", "straight_chair.n.01", etc,
    "category": a category such as "chair", "table", "building", "person", "airplane", "car", "seashell", "fish", etc. Try to be more specific than "furniture", but possibly more generic than with the synset,
    "width": approximate width in cm. For a human being this could be "45",
    "depth": approximate depth in cm. For a human being this could be "25",
    "height": approximate height in cm. For a human being this could be "175",
    "volume": approximate volume in l. For a human being this could be "62",
    "materials": a Python list of the materials that the object appears to be made of, taking into account the visible exterior and also likely interior (roughly in order of most used material to least used; include "air" if the object interior doesn't seem completely solid),
    "composition": a Python list with the apparent volume mixture of the materials above (make the list sum to 1),
    "mass": approximate mass in kilogram considering typical densities for the materials. For a human being this could be "72",
    "receptacle": a boolean indicating whether or not this object is a receptacle (e.g. a bowl, a cup, a vase, a box, a bag, etc). Return true or false with no explanations,
    "frontView": which of the views represents the front of the object (value should be the integer index associated with the chosen view). Note that the front view of an object, including furniture, tends to be the view that exhibits the highest degree of symmetry and detail, and it's usually the one you'd expect to observe when using the object,
    "onCeiling": whether this object can appear on the ceiling; return true or false with no explanations. This would be true for a ceiling fan but false for a chair,
    "onWall": whether this object can appear on the wall; return true or false with no explanations. This would be true for a painting but false for a table,
    "onFloor": whether this object can appear on the floor; return true or false with no explanations. This would be true for a piano but false for a curtain,
    "onObject": whether this object can appear on another object; return true or false with no explanations. This would be true for a table lamp but not for a sofa,
}
Please output your answer in the above JSON format.
"""

DEFAULT_QUESTION_NO_SYNSET = """Please annotate this 3D asset, corresponding to an object that can be found in a home, with the following values (output valid JSON, without additional comments):
"annotations": {
    "description": a description of the object (don't use the term "3D asset" or similar here),
    "category": a category such as "chair", "table", "building", "person", "airplane", "car", "seashell", "fish", "toy", etc. Be concise but specific, e.g. do not say "furniture" when "eames chair" would be more specific,
    "width": approximate width in cm. For a human being this could be "45",
    "depth": approximate depth in cm. For a human being this could be "25",
    "height": approximate height in cm. For a human being this could be "175",
    "volume": approximate volume in l. For a human being this could be "62",
    "materials": a Python list of the materials that the object appears to be made of, taking into account the visible exterior and also likely interior (roughly in order of most used material to least used; include "air" if the object interior doesn't seem completely solid),
    "composition": a Python list with the apparent volume mixture of the materials above (make the list sum to 1),
    "mass": approximate mass in kilogram considering typical densities for the materials. For a human being this could be "72",
    "receptacle": a boolean indicating whether or not this object is a receptacle (e.g. a bowl, a cup, a vase, a box, a bag, etc). Return true or false with no explanations,
    "frontView": which of the views represents the front of the object (value should be the integer index associated with the chosen view). Note that the front view of an object, including furniture, tends to be the view that exhibits the highest degree of symmetry and detail, and it's usually the one you'd expect to observe when using the object,
    "onCeiling": whether this object can appear on the ceiling; return true or false with no explanations. This would be true for a ceiling fan but false for a chair,
    "onWall": whether this object can appear on the wall; return true or false with no explanations. This would be true for a painting but false for a table,
    "onFloor": whether this object can appear on the floor; return true or false with no explanations. This would be true for a piano but false for a curtain,
    "onObject": whether this object can appear on another object; return true or false with no explanations. This would be true for a table lamp but not for a sofa,
}
Please output your answer in the above JSON format.
"""


def get_thumbnail_urls(
    uid: str,
    base_url: str = DEFAULT_THUMBNAIL_SOURCE_URL,
    view_indices: Sequence[str] = DEFAULT_VIEW_INDICES,
    local_renders: bool = False,
) -> List[Tuple[int, str]]:
    thumbnail_tuples = []

    for view_num, image_idx in enumerate(view_indices):
        if local_renders:
            fname = os.path.join(base_url, uid, f"render_{image_idx}.jpg")
            if not os.path.exists(fname):
                fname = os.path.join(base_url, f"render_{image_idx}.jpg")

            if os.path.isfile(fname):
                thumbnail_tuples.append((view_num, f"file://{fname}"))
            else:
                raise ValueError(f"Missing {fname}")
        else:
            url = f"{base_url}/{uid}/{str(image_idx).zfill(3)}.jpg"  # .zfill(3) ensures the number is three digits
            response = requests.head(url)  # HEAD request is faster than GET
            if response.status_code == 200:  # HTTP status code 200 means the URL exists
                thumbnail_tuples.append((view_num, url))
            else:
                raise ValueError(f"Unreachable {url}")

    return thumbnail_tuples


class GPTDialogue(TypedDict):
    prompt: List[Text]
    dialog: List[ComposedMessage]
    model: str


def describe_asset_from_views(
    uid: str,
    question: str = DEFAULT_QUESTION_NO_SYNSET,
    thumbnail_urls_cfg: Dict[str, Any] = None,
    **gpt_kwargs: Any,
) -> Tuple[str, List[str], GPTDialogue]:
    if thumbnail_urls_cfg is None:
        thumbnail_urls_cfg = {}

    # Get the urls of the available views.
    thumbnail_tuples = get_thumbnail_urls(uid, **thumbnail_urls_cfg)

    # Construct the initial prompt message. For the system description, we're using the
    # default content used in the OpenAI API document.
    prompt = Text(
        "You are ChatGPT, a large language model trained by OpenAI capable of looking at images."
        "\nCurrent date: 2023-03-05\nKnowledge cutoff: 2022-02\nImage Capabilities: Enabled",
        role="system",
    )

    user_messages = [
        Text(
            f"Here are {len(thumbnail_tuples)} views of a 3D asset."
            f" The series of images show the same asset from rotated views, so that you can see all sides of it.",
        )
    ]

    # Add each of the 3D asset's thumbnails to the conversation.
    urls = []
    for num, url in thumbnail_tuples:
        urls.append(url)

        if url.startswith("file://"):
            with open(url.replace("file://", ""), "rb") as f:
                buf = BytesIO(f.read())
                buf.seek(0)
            img_msg_contents = buf
        else:
            img_msg_contents = url

        user_messages.extend(
            [
                Text(f"View {num}"),
                Image(img_msg_contents),
            ]
        )

    # Finally, ask the question.
    user_messages.append(Text(question))

    all_gpt_kwargs = GPTDialogue(
        prompt=[prompt],
        dialog=[ComposedMessage(user_messages)],
        model="gpt-4-vision-preview",
    )
    all_gpt_kwargs.update(gpt_kwargs)

    answer = get_answer(**all_gpt_kwargs)

    return answer, urls, all_gpt_kwargs


def clean_up_json(json_string):
    """
    Tries to cleanup JSON using GPT.  If successful, it returns a valid JSON string.
    If it fails, it returns None.
    """
    prompt = Text(
        role="system",
        content="Convert this string into valid JSON, removing any unnecessary surrounding text before or after the JSON."
        " Ensure the JSON represents a dictionary with an 'annotations' key and a dictionary value"
        " containing all relevant data",
    )
    params = {
        "model": "gpt-4",
        "max_tokens": 2000,
    }
    json_string = get_answer([prompt], [Text(json_string)], **params)
    try:
        json.loads(json_string)
        return json_string
    except:
        return None


def get_initial_annotation(
    uid: str, get_best_synset: bool = True, **describe_kwargs: Any
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    json_str, urls, dialogue_dict = describe_asset_from_views(uid, **describe_kwargs)

    if json_str.startswith("```json"):
        json_str = json_str.replace("```json", "").replace("```", "")

    try:
        annotation = json.loads(json_str)
        assert "annotations" in annotation, f"Got annotation: {annotation}"
    except Exception as e:
        new_json_str = clean_up_json(json_str)
        if new_json_str is None:
            print(
                f"Failed to clean up response\n{json_str}\nwith urls\n{urls}\nwith error\n{e}"
            )
            raise
        annotation = json.loads(new_json_str)

    annotation = annotation["annotations"]
    annotation["uid"] = uid
    if get_best_synset:
        near_synsets, distances = nearest_synsets_from_annotation(
            annotation,
            save_to_dir=DESCRIPTION_EMBEDDING_OUTPUT_DIR,
            n_neighbors=NUM_NEIGHS,
        )
        annotation["near_synsets"] = {s: d for s, d in zip(near_synsets, distances)}

        for s in synsets_from_text(annotation["category"]):
            if s not in near_synsets:
                near_synsets.append(s)

        dialogue_dict["dialog"][0].content.append(
            Text(content=json_str, role="assistant")
        )
        dialogue_dict["dialog"][0].content.append(
            Text(content=prompt_for_best_synset(near_synsets), role="user")
        )

        answer = get_answer(**dialogue_dict).strip().lower()

        if answer.startswith("synset id: "):
            answer = answer.replace("synset id: ", "").strip()

        assert answer in near_synsets, f"Got answer {answer} not in {near_synsets}"

        annotation["synset"] = answer

    return annotation, urls
