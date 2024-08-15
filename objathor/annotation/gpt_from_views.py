import ast
import json
import os.path
import traceback
from io import BytesIO
from json import JSONDecodeError
from typing import List, Tuple, Sequence, Any, Optional, Dict, TypedDict

import requests

from objathor.annotation.synset_from_description import (
    nearest_synsets_from_annotation,
    DESCRIPTION_EMBEDDING_OUTPUT_DIR,
    NUM_NEIGHS,
    synsets_from_text,
    PICK_SINGLE_SYNSET_USING_OBJECT_INFO_TEMPLATE,
    synset_to_summary_str,
)
from objathor.constants import VISION_LLM, TEXT_LLM
from objathor.utils.gpt_utils import get_answer
from objathor.utils.queries import (
    Text,
    Image,
    ComposedMessage,
)

# Attribution: code adapted and extended from original by Yue Yang, developed during an internship at AI2

DEFAULT_THUMBNAIL_SOURCE_URL = "https://objaverse-im.s3.us-west-2.amazonaws.com"
DEFAULT_VIEW_INDICES = ("0", "3", "6", "9")

DEFAULT_QUESTION_NO_SYNSET = """Annotate this 3D asset assuming it can be found in an indoor environment (ie a home, a garage, an office, etc), with the following values:
"annotations": {
    "description_long": a very detailed visual description of the object that is no more than 6 sentences. Don't use the term "3D asset" or similar here and don't comment on the object's orientation. Do use proper nouns when appropriate.,
    "description": a 1-2 summary of description_long, keep the description rich and visual,
    "description_view_<i>": a short description of the object from view i (highlight/compare features that are different from other views),
    "category": a category such as "chair", "table", "building", "person", "airplane", "car", "seashell", "fish", "toy", etc. Be concise but specific, e.g. do not say "furniture" when "eames chair" would be more specific,
    "height": approximate height of the object in cm. Report the height for the object's orientation as shown in the images. For a standing human male this could be "175",
    "materials": a Python list of the materials that the object appears to be made of, taking into account the visible exterior and also likely interior (roughly in order of most used material to least used; include "air" if the object interior doesn't seem completely solid),
    "composition": a Python list with the apparent volume mixture of the materials above (make the list sum to 1),
    "mass": approximate mass in kilogram considering typical densities for the materials. For a human being this could be "72",
    "receptacle": a boolean indicating whether or not this object is a receptacle (e.g. a bowl, a cup, a vase, a box, a bag, etc). Return true or false with no explanations,
    "frontView": integer index of the view that represents the front of the object. This is typically the view from which you would approach the object to interact with it,
    "onCeiling": whether this object can appear on the ceiling; return true or false with no explanations. This would be true for a ceiling fan but false for a chair,
    "onWall": whether this object can appear on the wall; return true or false with no explanations. This would be true for a painting but false for a table,
    "onFloor": whether this object can appear on the floor; return true or false with no explanations. This would be true for a piano but false for a curtain,
    "onObject": whether this object can appear on another object; return true or false with no explanations. This would be true for a table lamp but not for a sofa,
    "quality": a number, 0-10, indicating the quality of the object. 0 is very low quality (amateurish, confusing, missing textures, a 3D scan with many holes, etc), 10 is very high quality (professional, detailed, etc).,
}
Output your answer in the above JSON format with NO OTHER TEXT.
"""

_lines = DEFAULT_QUESTION_NO_SYNSET.split("\n")
DEFAULT_QUESTION = "\n".join(
    _lines[:3]
    + [
        """    "synset": the synset of the object that is most closely related. Try to be as specific as possible. This could be "cat.n.01", "glass.n.03", "bank.n.02", "straight_chair.n.01", etc,"""
    ]
    + _lines[3:]
)

DEFAULT_QUESTION_THOR_ASSET_NO_MASS = """You are an expert in object annotation. Given images corresponding to an object that can be found in a home, you should annotate it by outputting the following JSON (without additional comments):
"annotations": {{
    "description_long": a very detailed visual description of the object that is no more than 6 sentences. Don't use the term "3D asset" or similar here and don't comment on the object's orientation. Do use proper nouns when appropriate.,
    "description": a 1-2 summary of description_long, keep the description rich and visual,
    "materials": a Python list of the materials that the object appears to be made of, taking into account the visible exterior and also likely interior (roughly in order of most used material to least used; include "air" if the object interior doesn't seem completely solid),
    "composition": a Python list with the apparent volume mixture of the materials above (make the list sum to 1),
    "onCeiling": whether this object can appear on the ceiling; return true or false with no explanations. This would be true for a ceiling fan but false for a chair,
    "onWall": whether this object can appear on the wall; return true or false with no explanations. This would be true for a painting but false for a table,
    "onFloor": whether this object can appear on the floor; return true or false with no explanations. This would be true for a piano but false for a curtain,
    "onObject": whether this object can appear on another object; return true or false with no explanations. This would be true for a table lamp but not for a sofa,
}}
Output your answer in the above JSON format WITH NO OTHER TEXT.
"""

_lines = DEFAULT_QUESTION_THOR_ASSET_NO_MASS.split("\n")
DEFAULT_QUESTION_THOR_ASSET = "\n".join(
    _lines[:5]
    + [
        """    "mass": approximate mass in kilogram considering typical densities for the materials. For a human being this could be "72","""
    ]
    + _lines[5:]
)

DEFAULT_OPENAI_VISION_PROMPT = (
    "You are an expert in 3D asset annotation. When providing annotations for 3D assets,"
    " you always treat the object as if it were real and in front of you."
)


DEFAULT_OPENAI_SYNSET_PROMPT = (
    "You are an expert in lexical categorization and on the WordNet database. When asked, provide expert"
    " feedback on the most appropriate synset for a given description."
)


def get_blender_render_urls(
    uid: str,
    local_renders: bool,
    base_url: str = DEFAULT_THUMBNAIL_SOURCE_URL,
    view_indices: Sequence[str] = DEFAULT_VIEW_INDICES,
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


def get_gpt_dialogue_to_describe_asset_from_views(
    uid: str,
    question: str = DEFAULT_QUESTION_NO_SYNSET,
    thumbnail_urls_cfg: Dict[str, Any] = None,
    thumbnail_urls: List[Tuple[int, str]] = None,
    extra_user_info: str = "",
    **gpt_kwargs: Any,
) -> Tuple[List[str], GPTDialogue]:
    assert (thumbnail_urls is None) != (
        thumbnail_urls_cfg is None
    ), "Either thumbnail_urls or thumbnail_urls_cfg must be provided, but not both."

    if thumbnail_urls_cfg is not None:
        # Get the urls of the available views.
        thumbnail_tuples = get_blender_render_urls(uid, **thumbnail_urls_cfg)
    else:
        thumbnail_tuples = []
        for i, p in thumbnail_urls:
            if os.path.exists(p):
                thumbnail_tuples.append((i, f"file://{p}"))
            else:
                thumbnail_tuples.append((i, p))

    # Construct the initial prompt message. For the system description, we're using the
    # default content used in the OpenAI API document.
    prompt = Text(
        DEFAULT_OPENAI_VISION_PROMPT,
        role="system",
    )

    dialogue = [
        Text(
            f"Here are {len(thumbnail_tuples)} views of a 3D asset."
            f" The series of images show the same asset from rotated views,"
            f" so that you can see all sides of it.{extra_user_info}",
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

        dialogue.extend(
            [
                Text(f"View {num}"),
                Image(img_msg_contents),
            ]
        )

    # Finally, ask the question.
    dialogue.append(Text(question))

    all_gpt_kwargs = GPTDialogue(
        prompt=[prompt],
        dialog=[ComposedMessage(dialogue)],
        model=VISION_LLM,
    )
    all_gpt_kwargs.update(gpt_kwargs)
    return urls, all_gpt_kwargs


def describe_asset_from_views(
    uid: str,
    question: str = DEFAULT_QUESTION_NO_SYNSET,
    thumbnail_urls_cfg: Dict[str, Any] = None,
    thumbnail_urls: List[Tuple[int, str]] = None,
    extra_user_info: str = "",
    **gpt_kwargs: Any,
) -> Tuple[str, List[str], GPTDialogue]:
    urls, all_gpt_kwargs = get_gpt_dialogue_to_describe_asset_from_views(
        uid=uid,
        question=question,
        thumbnail_urls_cfg=thumbnail_urls_cfg,
        thumbnail_urls=thumbnail_urls,
        extra_user_info=extra_user_info,
        **gpt_kwargs,
    )

    answer = get_answer(**all_gpt_kwargs)

    return answer, urls, all_gpt_kwargs


def gpt_dialogue_to_batchable_request(gpt_dialogue: GPTDialogue) -> Dict[str, Any]:

    messages = [
        dict(role=msg.role, content=[msg.gpt()]) for msg in gpt_dialogue["prompt"]
    ] + [dict(role=msg.role, content=msg.gpt()) for msg in gpt_dialogue["dialog"]]

    return (
        # dict(
        # custom_id=uid,
        # method="POST",
        # url="/v1/chat/completions",
        # body=
        dict(
            model=gpt_dialogue["model"],
            messages=messages,
            max_tokens=2000,
            temperature=0.0,
        )
    )


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
        "model": TEXT_LLM,
        "max_tokens": 2000,
    }
    json_string = get_answer([prompt], [Text(json_string)], **params)
    try:
        json.loads(json_string)
        return json_string
    except JSONDecodeError:
        return None


def get_gpt_dialogue_to_get_best_synset_using_annotations(
    annotation: Dict[str, Any],
    n_neighbors: int = NUM_NEIGHS,
    **chat_kwargs: Any,
) -> GPTDialogue:

    near_synsets, distances = nearest_synsets_from_annotation(
        annotation,
        save_to_dir=DESCRIPTION_EMBEDDING_OUTPUT_DIR,
        n_neighbors=n_neighbors,
    )
    distances = distances.tolist()
    for s in synsets_from_text(annotation["category"]):
        if s not in near_synsets:
            near_synsets.append(s)
            distances.append(-1000.0)

    annotation["near_synsets"] = {s: d for s, d in zip(near_synsets, distances)}

    # Construct the initial prompt message. For the system description, we're using the
    # default content used in the OpenAI API document.
    prompt = Text(
        DEFAULT_OPENAI_SYNSET_PROMPT,
        role="system",
    )

    user_messages = [
        Text(
            PICK_SINGLE_SYNSET_USING_OBJECT_INFO_TEMPLATE.format(
                description=f"A {annotation['category']}. {annotation['description']}",
                scale=float(annotation["height"]) * 0.01,
            )
            + "\n"
            + "\n\n".join([synset_to_summary_str(s) for s in near_synsets]),
            role="user",
        )
    ]

    dialogue_dict = chat_kwargs

    dialogue_dict["prompt"] = [prompt]
    dialogue_dict["dialog"] = [ComposedMessage(user_messages)]
    dialogue_dict["model"] = TEXT_LLM
    return dialogue_dict


def get_best_synset_using_annotations(
    annotation: Dict[str, Any],
    n_neighbors: int = NUM_NEIGHS,
    retry: bool = False,
    **chat_kwargs,
) -> str:
    dialogue_dict = get_gpt_dialogue_to_get_best_synset_using_annotations(
        annotation=annotation,
        n_neighbors=n_neighbors,
        **chat_kwargs,
    )
    near_synsets = list(annotation["near_synsets"].keys())

    answer = get_answer(**dialogue_dict).strip().lower()

    if answer.startswith("synset id: "):
        answer = answer.replace("synset id: ", "").strip()

    if retry and (answer not in near_synsets):
        print(
            f"Got answer {answer} not in {near_synsets}. Retrying with n_neighbors = {n_neighbors * 2}",
            flush=True,
        )
        return get_best_synset_using_annotations(
            annotation=annotation,
            n_neighbors=n_neighbors * 2,
            retry=False,
            **chat_kwargs,
        )

    assert answer in near_synsets, f"Got answer {answer} not in {near_synsets}"

    return answer


def load_gpt_annotations_from_json_str(
    uid: str, json_str: str, attempt_cleanup: bool
) -> Dict[str, Any]:
    if json_str.startswith("```json"):
        json_str = json_str.replace("```json", "").replace("```", "")

    try:
        annotation = json.loads(json_str)
    except JSONDecodeError:
        annotation = None
        try:
            annotation = ast.literal_eval(json_str)
        except:
            pass

        if annotation is None:
            new_json_str = None
            if attempt_cleanup:
                new_json_str = clean_up_json(json_str)

            if new_json_str is None:
                print(
                    f"Failed to clean up response"
                    f"\n{json_str}"
                    f"\nwith error:"
                    f"\n{traceback.format_exc()}"
                )
                raise

            annotation = json.loads(new_json_str)

    if "annotations" in annotation:
        annotation = annotation["annotations"]
    elif "annotation" in annotation:
        annotation = annotation["annotation"]

    assert "description" in annotation, f"Missing description in {annotation}"
    annotation["uid"] = uid
    return annotation


def get_initial_annotation(
    uid: str, get_best_synset: bool = True, **describe_kwargs: Any
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    json_str, urls, dialogue_dict = describe_asset_from_views(uid, **describe_kwargs)

    annotation = load_gpt_annotations_from_json_str(
        uid=uid, json_str=json_str, attempt_cleanup=True
    )

    if get_best_synset:
        try:
            annotation["synset"] = get_best_synset_using_annotations(
                annotation,
                **dialogue_dict,
            )
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            print(
                f"[ERROR] Failed to get best synset using annotations for uid {uid}. Annotations"
                f" are {annotation}",
                flush=True,
            )
            raise

    if "synset" in annotation:
        annotation["wn_version"] = "oewn:2022"

    return annotation, urls
