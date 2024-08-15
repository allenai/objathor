from typing import TypedDict, List, Dict

MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS = {
    # OpenAI models
    "gpt-3.5-turbo-0301": 1.5,
    "gpt-3.5-turbo-0125": 1.5,
    "gpt-4-1106-preview": 10.0,
    "gpt-4o-2024-05-13": 5.0,
    "gpt-4o-2024-08-06": 2.5,
    "gpt-4o-mini-2024-07-18": 0.15,
}

MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS = {
    # OpenAI models
    "gpt-3.5-turbo-0301": 2.0,
    "gpt-3.5-turbo-0125": 2.0,
    "gpt-4-1106-preview": 30.0,
    "gpt-4o-2024-05-13": 15.0,
    "gpt-4o-2024-08-06": 10.0,
    "gpt-4o-mini-2024-07-18": 0.6,
}


class LicenseInfo(TypedDict):
    """
    A TypedDict representing the license information of an object.
    See the Objaverse paper for more details: https://arxiv.org/abs/2307.05663

    Attributes:
        license (str): The type of license.
        uri (str): The URI of the object.
        creator_username (str): The username of the creator.
        creator_display_name (str): The display name of the creator.
        creator_profile_url (str): The profile URL of the creator.
    """

    license: str
    uri: str
    creator_username: str
    creator_display_name: str
    creator_profile_url: str


class ObjectAnnotation(TypedDict):
    """
    A TypedDict representing the annotation of an object. Note that
    some of these fields may be missing in general.

    Attributes:
        description (str): The description of the object.
        category (str): The category of the object (as a simple string).
        width (float): The width of the object in cm.
        depth (float): The depth of the object in cm.
        height (float): The height of the object in cm.
        volume (float): The volume of the object in l.
        materials (List[str]): The materials that the object appears to be made of.
        composition (List[float]): Relative composition of each material by percentage.
        mass (float): The mass of the object in kg.
        receptacle (bool): Whether the object is a receptacle.
        frontView (int): The view that represents the front of the object.
        onCeiling (bool): Whether the object can appear on the ceiling.
        onWall (bool): Whether the object can appear on a wall.
        onFloor (bool): Whether the object can appear on the floor.
        onObject (bool): Whether the object can appear on another object.
        uid (str): The unique identifier of the object.
        near_synsets (Dict[str, float]): Related synsets with similarity scores.
        synset (str): The synset of the object.
        wn_version (str): Version of WordNet used for synset identification. E.g. Open English WordNet 2022 == 'oewn:2022'
        pose_z_rot_angle (float): Rotation angle around the Z axis in radians.
        scale (float): The scale of the object. Possible redundant if width/depth/height are already given.
        z_axis_scale (bool): Whether `scale` corresponds to the z-axis or the max size across all x/y/z dimensions.
        license_info (LicenseInfo): The license information of the object.

    Example:
       {
           "description": "This is a collectible sports card encased in a protective slab, featuring a printed image and information about a baseball player.",
           "category": "sports card",
           "width": 7.6,
           "depth": 0.5,
           "height": 10.2,
           "volume": 0.03876,
           "materials": ["plastic", "paper", "ink", "air"],
           "composition": [0.5, 0.3, 0.1, 0.1],
           "mass": 0.05,
           "receptacle": false,
           "frontView": 3,
           "onCeiling": false,
           "onWall": true,
           "onFloor": false,
           "onObject": true,
           "uid": "80c7c462949740c180255852fe0e8079",
           "near_synsets": {
               "baseball_card.n.01": 0.4129518901204248,
               "trading_card.n.01": 0.5455496005249564,
               "baseball.n.02": 0.5610857997176745,
               "card.n.04": 0.5729855006462443,
               "picture_postcard.n.01": 0.5773243861625321,
           },
           "synset": "baseball_card.n.01",
           "wn_version": "oewn:2022",
           "pose_z_rot_angle": 4.71238898038469,
           "scale": 0.102,
           "z_axis_scale": true,
           "license_info": {
               "license": "by",
               "uri": "https://api.sketchfab.com/v3/models/80c7c462949740c180255852fe0e8079",
               "creator_username": "GeneralCreed",
               "creator_display_name": "GeneralCreed",
               "creator_profile_url": "https://sketchfab.com/GeneralCreed",
           },
       }
    """

    description: str
    category: str
    width: float
    depth: float
    height: float
    volume: float
    materials: List[str]
    composition: List[float]
    mass: float
    receptacle: bool
    frontView: int
    onCeiling: bool
    onWall: bool
    onFloor: bool
    onObject: bool
    uid: str
    near_synsets: Dict[str, float]
    synset: str
    wn_version: str
    pose_z_rot_angle: float
    scale: float
    z_axis_scale: bool
    license_info: LicenseInfo


def compute_llm_cost(input_tokens: int, output_tokens: int, model: str):
    assert (
        model in MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS
        and model in MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS
    ), f"model [{model}] must be in both MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS and MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS"

    input_token_cost_per_1m = MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS[model]
    output_token_cost_per_1m = MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS[model]

    return (
        input_tokens * input_token_cost_per_1m
        + output_tokens * output_token_cost_per_1m
    ) / 1e6
