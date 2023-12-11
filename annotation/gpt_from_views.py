from typing import List, Tuple, Sequence, Any, Optional, Dict
import json

import requests

from utils.gpt_utils import get_answer


DEFAULT_THUMBNAIL_SOURCE_URL = "https://objaverse-im.s3.us-west-2.amazonaws.com"
DEFAULT_VIEW_INDICES = ("0", "3", "6", "9")

DEFAULT_QUESTION = """Please annotate this 3D asset with the following values (output valid JSON):
"annotations": {
    "category": a category such as "chair", "table", "building", "person", "airplane", "car", "seashell", "fish", etc. Try to be more specific than "furniture",
    "width": approximate width in cm. For a human being this could be "45",
    "length": approximate length in cm. For a human being this could be "25",
    "height": approximate height in cm. For a human being this could be "175",
    "frontView": which of the views represents the front of the object (value should be the integer index associated with the chosen view). Note that the front view of an object, including furniture, tends to be the view that exhibits the highest degree of symmetry,
    "description": a description of the object (don't use the term "3D asset" or similar here),
    "materials": a Python list of the materials that the object appears to be made of (roughly in order of most used material to least used),
    "composition": a Python list with the apparent mixture of the materials listed above (make the list sum to 1),
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
) -> List[Tuple[int, str]]:
    thumbnail_tuples = []

    for view_num, image_idx in enumerate(view_indices):
        url = f"{base_url}/{uid}/{str(image_idx).zfill(3)}.png"  # .zfill(3) ensures the number is three digits
        response = requests.head(url)  # HEAD request is faster than GET
        if response.status_code == 200:  # HTTP status code 200 means the URL exists
            thumbnail_tuples.append((view_num, url))

    return thumbnail_tuples


def describe_asset_from_views(
    uid: str, question: str = DEFAULT_QUESTION, **thumbnail_urls_cfg: Any
) -> Tuple[str, List[str]]:
    # Get the urls of the available views.
    thumbnail_tuples = get_thumbnail_urls(uid, **thumbnail_urls_cfg)

    # Construct the initial prompt message. For the system description, we're using the
    # default content used in the OpenAI API document.
    prompt = [
        dict(
            role="system",
            content=[
                dict(
                    type="text",
                    text="You are ChatGPT, a large language model trained by OpenAI capable of looking at images."
                    "\nCurrent date: 2023-03-05\nKnowledge cutoff: 2022-02\nImage Capabilities: Enabled",
                )
            ],
        )
    ]

    user_messages = [
        dict(
            type="text",
            text=f"Here are {len(thumbnail_tuples)} views of a 3D asset."
            f" The series of images show the same asset from rotated views, so that you can see all sides of it.",
        )
    ]

    # Add each of the 3D asset's thumbnails to the conversation.
    urls = []
    for num, url in thumbnail_tuples:
        urls.append(url)
        user_messages.extend(
            [
                dict(type="text", text=f"View {num}"),
                dict(type="image_url", image_url=dict(url=url, detail="high")),
            ]
        )

    # Finally, ask the question.
    user_messages.append(dict(type="text", text=question))

    answer = get_answer(
        prompt=prompt,
        query=user_messages,
        model="gpt-4-vision-preview",
    )

    return answer, urls


def get_initial_annotation(
    uid: str, **describe_kwargs: Any
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    json_str, urls = describe_asset_from_views(uid, **describe_kwargs)

    if json_str.startswith("```json"):
        json_str = json_str.replace("```json", "").replace("```", "")

    annotation = json.loads(json_str)
    assert "annotations" in annotation, f"Got annotation: {annotation}"

    return annotation["annotations"], urls
