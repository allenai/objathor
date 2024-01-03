from importlib import import_module
from typing import Union, Callable, Any, Dict, Optional
import argparse
import json

import compress_pickle
import compress_json

from objathor.annotation.gpt_from_views import get_initial_annotation
from objathor.utils.blender import render_glb_from_angles


def write(
    anno: Dict[str, Any],
    output_file: Union[str, Callable[[Dict[str, Any]], Optional[Any]]],
    **kwargs: Any,
) -> None:
    if isinstance(output_file, str):
        if output_file.endswith(".json.gz"):
            compress_json.dump(anno, output_file, json_kwargs=dict(indent=2))
        elif output_file.endswith(".pickle.gz") or output_file.endswith(".pkl.gz"):
            compress_pickle.dump(anno, output_file)
        else:
            try:
                module_name, function_name = output_file.rsplit(".", 1)
                getattr(import_module(module_name), function_name)(anno, **kwargs)
            except Exception as e:
                print("Error", e)
                raise NotImplementedError(
                    "Only .pkl.gz and .json.gz supported, besides appropriate library function identifiers"
                )
    elif isinstance(output_file, Callable):
        output_file(anno)
    else:
        raise NotImplementedError(
            f"Unsupported output_file arg of type {type(output_file).__name__}"
        )


def annotate_asset(
    uid: str,
    output_file: Union[str, Callable[[Dict[str, Any]], bool]],
    local_render: bool = False,
    **kwargs: Any,
) -> None:
    if local_render:
        base_dir = "/tmp/objathor"

        angles = [0, 90, 180, 270]
        render_glb_from_angles(uid, base_dir, angles=angles)

        anno, urls = get_initial_annotation(
            uid,
            thumbnail_urls_cfg=dict(
                base_url=base_dir,
                view_indices=[str(float(angle)) for angle in angles],
                local_renders=True,
            ),
        )
        anno["pose_z_rot_angle"] = angles[anno["frontView"]]
    else:
        anno, urls = get_initial_annotation(uid)

    anno["scale"] = float(anno["height"]) / 100
    anno["z_axis_scale"] = True

    anno["pre_rendered_views_urls"] = urls
    anno["uid"] = uid
    write(anno, output_file, **kwargs)


def parse_args(
    description="Generate GPT-based annotation of an objaverse asset and save to disk",
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--uid",
        type=str,
        required=True,
        help="The UID of the pre-rendered Objaverse asset to annotate",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output file ('.pkl.gz' and '.json.gz' supported) to write to, or library function to be called",
    )
    parser.add_argument(
        "--output_func_kwargs",
        type=str,
        required=False,
        default=None,
        help="Additional arguments for library method output as a json dict",
    )
    parser.add_argument(
        "--local_render",
        action="store_true",
        help="Generate object views to be uploaded to GPT locally (requires blender)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_func_kwargs is None:
        output_func_kwargs = {}
    else:
        output_func_kwargs = json.loads(args.output_func_kwargs)

    annotate_asset(
        args.uid, args.output, local_render=args.local_render, **output_func_kwargs
    )


if __name__ == "__main__":
    main()
