import pytest


def test_dummy():
    assert True

#
# import json
# import os
# import shutil
# from sys import platform

# import pytest

# import objathor.asset_conversion.util as util
# from objathor.asset_conversion.pipeline_to_thor import main as pipeline_main


# def run_pipeline_main(object_id, out_path, extension, annotation=None):
#     annotation = annotation or {
#         object_id: {
#             "category": "flip-flop (sandal)",
#             "category_max_scale": 0.4,
#             "category_min_scale": 0.25,
#             "description": "A black sandal with grey stripes.",
#             "pose_z_rot_angle": 3.67,
#             "ref_category": "Boots",
#             "scale": 0.32,
#             "uid": object_id,
#             "z_axis_scale": False,
#         }
#     }
#     annotation_filename = "annotation.json"
#     os.makedirs(out_path, exist_ok=True)
#     annotation_path = ""
#     if annotation:
#         annotation_path = os.path.abspath(os.path.join(out_path, annotation_filename))
#         with open(annotation_path, "w") as f:
#             json.dump(annotation, f)
#     if platform == "darwin":
#         thor_platform = f"OSXIntel64"
#     else:
#         # TODO distinguish intel vs M2
#         thor_platform = f"CloudRendering"

#     return pipeline_main(
#         [
#             # f"/Users/alvaroh/ai2/objathor/objathor/asset_conversion/pipeline_to_thor.py",
#             f"--uids={object_id}",
#             f"--output_dir={out_path}",
#             "--live",
#             f"--extension={extension}",
#             "--blender_as_module",
#             f"--annotations={annotation_path}",
#             f"--thor_platform={thor_platform}",
#             # TODO remove when vulkan is properly configured in CI, not available through github actions
#             f"--skip_thor_metadata",
#         ]
#     )


# @pytest.mark.skipif(
#     platform == "linux" or platform == "linux2", reason="Dependency on lib"
# )
# def test_pipeline_to_thor_w_annotation():
#     object_id = "000074a334c541878360457c672b6c2e"
#     annotation = {
#         object_id: {
#             "category": "flip-flop (sandal)",
#             "category_max_scale": 0.4,
#             "category_min_scale": 0.25,
#             "description": "A black sandal with grey stripes.",
#             "pose_z_rot_angle": 3.67,
#             "ref_category": "Boots",
#             "scale": 0.32,
#             "uid": object_id,
#             "z_axis_scale": False,
#         }
#     }
#     out_path = os.path.abspath(os.path.join(".", "test-out"))
#     extension = ".json"

#     result = run_pipeline_main(
#         object_id=object_id,
#         out_path=out_path,
#         extension=extension,
#         annotation=annotation,
#     )

#     assert util.get_existing_thor_asset_file_path(
#         out_dir=os.path.join(out_path, object_id), asset_id=object_id
#     ) == os.path.join(out_path, object_id, f"{object_id}{extension}")
#     shutil.rmtree(out_path)


# @pytest.mark.skipif(
#     platform == "linux" or platform == "linux2", reason="Dependency on lib"
# )
# def test_pipeline_to_thor_msgpack_gz():
#     object_id = "000074a334c541878360457c672b6c2e"
#     out_path = os.path.abspath(os.path.join(".", "test-out"))
#     extension = ".msgpack.gz"
#     result = run_pipeline_main(
#         object_id=object_id, out_path=out_path, extension=extension
#     )

#     assert util.get_existing_thor_asset_file_path(
#         out_dir=os.path.join(out_path, object_id), asset_id=object_id
#     ) == os.path.join(out_path, object_id, f"{object_id}{extension}")
#     shutil.rmtree(out_path)


# # Sum of tests make it too slow for CI
# @pytest.mark.skipif(
#     platform == "linux" or platform == "linux2", reason="Too slow for CI"
# )
# def test_pipeline_to_thor_msgpack():
#     object_id = "000074a334c541878360457c672b6c2e"
#     out_path = os.path.abspath(os.path.join(".", "test-out"))
#     extension = ".msgpack"
#     result = run_pipeline_main(
#         object_id=object_id, out_path=out_path, extension=extension
#     )

#     assert util.get_existing_thor_asset_file_path(
#         out_dir=os.path.join(out_path, object_id),
#         asset_id=object_id,
#         force_extension=extension,
#     ) == os.path.join(out_path, object_id, f"{object_id}{extension}")
#     shutil.rmtree(out_path)


# @pytest.mark.skipif(
#     platform == "linux" or platform == "linux2", reason="Too slow for CI"
# )
# def test_pipeline_to_thor_gz():
#     object_id = "000074a334c541878360457c672b6c2e"
#     out_path = os.path.abspath(os.path.join(".", "test-out"))
#     extension = "json.gz"
#     result = run_pipeline_main(
#         object_id=object_id, out_path=out_path, extension=extension
#     )

#     assert util.get_existing_thor_asset_file_path(
#         out_dir=os.path.join(out_path, object_id), asset_id=object_id
#     ) == os.path.join(out_path, object_id, f"{object_id}{extension}")
#     shutil.rmtree(out_path)


# @pytest.mark.skipif(
#     platform == "linux" or platform == "linux2", reason="Too slow for CI"
# )
# def test_pipeline_to_thor_pkl_gz():
#     object_id = "000074a334c541878360457c672b6c2e"
#     out_path = os.path.abspath(os.path.join(".", "test-out"))
#     extension = ".pkl.gz"
#     result = run_pipeline_main(
#         object_id=object_id, out_path=out_path, extension=extension
#     )

#     assert util.get_existing_thor_asset_file_path(
#         out_dir=os.path.join(out_path, object_id), asset_id=object_id
#     ) == os.path.join(out_path, object_id, f"{object_id}{extension}")
#     shutil.rmtree(out_path)
