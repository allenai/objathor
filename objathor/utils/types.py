import enum
from typing import TypedDict, Literal

PROCESSED_ASSET_EXTENSIONS = Literal[
    ".json", "json.gz", ".pkl.gz", ".msgpack", ".msgpack.gz"
]


class ObjathorStatus(enum.Enum):
    ANNOTATION_SUCCESS = "annotation_success"
    OPTIMIZATION_SUCCESS = "optimization_success"
    SUCCESS = "success"

    # Annotation
    JSON_DECODE_FAIL = "json_decode_fail"

    ## Annotation async

    #### Waiting
    ANNOTATE_VIEWS_IN_PROGRESS = "annotate_views_in_progress"
    SYNSET_IN_PROGRESS = "synset_in_progress"

    ### Async failures
    ASYNC_ANNOTATE_VIEWS_REQUEST_FAIL = "async_annotate_views_request_fail"
    ASYNC_SYNSET_REQUEST_FAIL = "async_synset_request_fail"

    # Blender failures
    BLENDER_RENDER_FAIL = "blender_render_fail"
    BLENDER_PROCESS_FAIL = "blender_process_fail"
    BLENDER_PROCESS_TIMEOUT_FAIL = "blender_process_timeout_fail"
    IMAGE_COMPRESS_FAIL = "png_to_jpg_compression_fail"
    GENERATE_COLLIDERS_FAIL = "vhacd_generate_colliders_fail"
    THOR_CREATE_ASSET_FAIL = "thor_create_asset_fail"
    THOR_VIEW_ASSET_FAIL = "thor_view_asset_in_thor_fail"
    THOR_PROCESS_FAIL = "thor_process_fail"

    # Unknown
    UNKNOWN_FAIL = "unknown_fail"

    def is_fail(self) -> bool:
        return self.value.endswith("fail")

    def is_success(self) -> bool:
        return self.value.endswith("success")

    def is_in_progress(self) -> bool:
        return self.value.endswith("in_progress")


for _e in ObjathorStatus:
    assert any(
        _e.value.endswith(k) for k in ["success", "fail", "in_progress"]
    ), f"PipelineStatus values should all end in 'success', 'fail', or 'in_progress'"


class ObjathorInfo(TypedDict):
    status: ObjathorStatus
