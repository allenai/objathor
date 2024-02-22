import argparse
import gzip
import json
import logging
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional

# TODO import shared libs, not sure how to find inside of blender
# from data_generation.asset_conversion.util import get_json_save_path

try:
    import bpy
    import bmesh
except ImportError as e:
    raise ImportError(
        f"{e}: Blender is not installed, make sure to either run 'pip install bpy' to install it as a module or as an application https://docs.blender.org/manual/en/latest/getting_started/installing/index.html"
    )

import pickle

import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path in sys.path:
    sys.path.append(dir_path)

import importlib
import util

importlib.reload(util)

FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger(__name__)


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def reset_scene():
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            # uv_layers = obj.data.uv_layers
            # for texture in uv_layers:
            #     try:
            #         uv_layers.remove(texture)
            #     except:
            #         logger.debug(obj)
            #         raise Exception()
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def purge_orphan_data():
    for block in bpy.data.collections:
        if block.users == 0:
            bpy.data.collections.remove(block)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def load_model(model_path: str) -> None:
    assert model_path.endswith(".glb")
    bpy.ops.import_scene.gltf(filepath=model_path, merge_vertices=True)


def flatten_scene_hierarchy():
    for obj in bpy.data.objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for modifier in obj.modifiers:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.parent_clear(type="CLEAR")


def delete_nonmesh_objects():
    # select non-mesh objects
    non_meshes = [obj for obj in bpy.data.objects if obj.type != "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    # select all objects for deletion
    for obj in non_meshes:
        obj.select_set(True)
    bpy.ops.object.delete()


def resize_object(mesh: bpy.types.Object, max_side_length_meters: float) -> None:
    # select the mesh
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    # get the bounding box
    x_size, y_size, z_size = mesh.dimensions
    # get the max side length
    curr_max_side_length = max([x_size, y_size, z_size])
    # get the scale factor
    scale_factor = max_side_length_meters / curr_max_side_length
    # scale the object
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
    # 0 out the transform
    bpy.ops.object.transforms_to_deltas(mode="ALL")


def center_mesh(mesh: bpy.types.Object) -> None:
    # select the mesh
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    # clear and keep the transformation of the parent
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    # set the mesh position to the origin, use the bounding box center
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.object.location = (0, 0, 0)
    # 0 out the transform
    bpy.ops.object.transforms_to_deltas(mode="ALL")


def join_meshes() -> bpy.types.Object:
    # get all the meshes in the scene
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    # join all of the meshes
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
    # join the meshes
    bpy.ops.object.join()
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    assert len(meshes) == 1
    mesh = meshes[0]
    return mesh


def is_mesh_open(mesh: bpy.types.Object) -> bool:
    bpy.ops.object.editmode_toggle()
    bm = bmesh.from_edit_mesh(mesh.data)

    # Clear the BMesh selection
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    # Look for boundary edges
    for edge in bm.edges:
        if edge.is_boundary:
            bpy.ops.object.editmode_toggle()
            return True
    # Write the BMesh back to the mesh and exit edit mode
    bpy.ops.object.editmode_toggle()
    return False

def get_min_decimation(mesh: bpy.ops.object) -> float:
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    bpy.ops.object.duplicate()
    dec_test_object = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = dec_test_object

    dec_mod_name = "Decimate_Test"
    bpy.ops.object.modifier_add(type="DECIMATE")
    bpy.context.object.modifiers[-1].name = dec_mod_name
    bpy.context.object.modifiers[dec_mod_name].ratio = 0
    bpy.context.object.modifiers[dec_mod_name].use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec_mod_name)
    dec_min = len(dec_test_object.data.polygons)
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action='DESELECT')

    return dec_min

def decimate(mesh: bpy.types.Object, decimation_ratio: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.modifier_add(type="DECIMATE")
    bpy.context.object.modifiers["Decimate"].ratio = decimation_ratio
    bpy.context.object.modifiers["Decimate"].use_collapse_triangulate = True

    try:
        bpy.ops.object.modifier_apply(modifier="Decimate")
    except:
        pass


def add_uvmap(
    mesh: bpy.types.Object, image_height: int, image_width: int, texture_path: str
) -> None:
    # select the mesh
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    # delete all existing uv maps
    # for uv_map in mesh.data.uv_layers:
    #     mesh.data.uv_layers.remove(uv_map)
    # create a new uv map
    # get all uv map names
    uv_map_names = [uv_map.name for uv_map in mesh.data.uv_layers]
    bpy.ops.mesh.uv_texture_add()
    # get the new uv map name
    new_uv_map_names = [uv_map.name for uv_map in mesh.data.uv_layers]
    new_uv_map_name = list(set(new_uv_map_names) - set(uv_map_names))[0]
    # set the active uv map to the new one
    # set the active rendered uv map to the new one
    # uv_layer_name = list(mesh.data.uv_layers.keys())[0]
    # mesh.data.uv_layers[uv_layer_name].active_render = True
    # open up the UV Editing tab
    bpy.ops.object.mode_set(mode="EDIT")
    # get the name of the new uv map
    bpy.ops.uv.smart_project(island_margin=0.002, area_weight=0)
    # create a new image of the uv map
    image = bpy.data.images.new(
        name=texture_path, height=image_height, width=image_width
    )
    return image, new_uv_map_name


def create_uv_map(object: bpy.types.Object, texture_size: int) -> None:
    island_separation = 0.001 / (texture_size / 512)
    logger.debug(f"ISLAND SEPARATION: {str(island_separation)}")

    object.select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    # Smart project method
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(30), island_margin=island_separation, area_weight=1.0
    )

    bpy.ops.object.mode_set(mode="OBJECT")


def process_material(
    material: bpy.types.Material,
    image: bpy.data.images,
) -> None:
    material.use_nodes = True
    # get the principled BSDF node
    bsdf = material.node_tree.nodes["Principled BSDF"]
    # create a new image texture nodeA
    image_texture = material.node_tree.nodes.new(type="ShaderNodeTexImage")
    image_texture.image = image
    # get the metallic input
    metallic_input = bsdf.inputs["Metallic"]
    # get the links
    links = material.node_tree.links
    # get the links that are connected to the metallic input
    connected_links = [link for link in links if link.to_socket == metallic_input]
    # break the connection
    for link in connected_links:
        links.remove(link)
    bsdf.inputs["Metallic"].default_value = 0
    # deselect all nodes in the node_tree
    for node in material.node_tree.nodes:
        node.select = False
    # select the image texture node
    # image_texture.select = True
    material.node_tree.nodes.active = image_texture


def set_material_uvs(image: bpy.types.Image) -> None:
    # add image texture to the material and set the image to the uv map
    throw_exception = False
    for material in bpy.data.materials:
        if "Principled BSDF" not in material.node_tree.nodes:
            bsdf = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            # connect the output of the principled BSDF node to the material output
            material.node_tree.links.new(
                bsdf.outputs["BSDF"],
                material.node_tree.nodes["Material Output"].inputs["Surface"],
            )
            if "Image Texture" in material.node_tree.nodes:
                image_texture = material.node_tree.nodes["Image Texture"]
                # connect the image texture to the base color of the principled BSDF node
                material.node_tree.links.new(
                    image_texture.outputs["Color"], bsdf.inputs["Base Color"]
                )
            throw_exception = True
    # if throw_exception:
    #     raise ValueError("Principled BSDF node not found in material")
    for material in bpy.data.materials:
        process_material(material, image)


def get_visibility_points(
    mesh: bpy.types.Object,
    voxel_size: float = 0.05,
    min_voxels: int = 3,
    visualize: bool = False,
    add_unity_rotation_correction: bool = False,
) -> List[Dict[str, float]]:
    """Get visibility points for a mesh.
    add_unity_rotation_correction: if True, add a rotation correction to the
        points so that they are in the same coordinate system as Unity. This
        rotates the points 90 degrees around the x-axis.
    """
    vertices = mesh.data.vertices
    # convert the vertices to a numpy array
    vertices = np.array(
        [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    )
    x_max, y_max, z_max = vertices.max(axis=0)
    x_min, y_min, z_min = vertices.min(axis=0)
    # get the voxels in each direction
    xs = np.linspace(
        x_min, x_max, max(min_voxels, int((x_max - x_min) / voxel_size)), endpoint=True
    )
    ys = np.linspace(
        y_min, y_max, max(min_voxels, int((y_max - y_min) / voxel_size)), endpoint=True
    )
    zs = np.linspace(
        z_min, z_max, max(min_voxels, int((z_max - z_min) / voxel_size)), endpoint=True
    )
    if len(xs) > 10:
        xs = np.linspace(
            x_min, x_max, int((x_max - x_min) / (voxel_size * 2)), endpoint=True
        )
    if len(ys) > 10:
        ys = np.linspace(
            y_min, y_max, int((y_max - y_min) / (voxel_size * 2)), endpoint=True
        )
    if len(zs) > 10:
        zs = np.linspace(
            z_min, z_max, int((z_max - z_min) / (voxel_size * 2)), endpoint=True
        )
    x_voxel_size = xs[1] - xs[0]
    y_voxel_size = ys[1] - ys[0]
    z_voxel_size = zs[1] - zs[0]
    logger.debug(
        (
            len(mesh.data.polygons),
            len(xs) * len(ys) * len(zs),
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            min_voxels,
            voxel_size,
            x_voxel_size,
            y_voxel_size,
            z_voxel_size,
        )
    )
    # sample points on the surface of the mesh
    if len(mesh.data.polygons) == 0:
        if len(mesh.data.vertices) == 0:
            return {}
        else:
            points = np.array(
                [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
            )
            points = [dict(x=x, y=y, z=z) for x, y, z in points.tolist()]
            num_points = len(xs) * len(ys) * len(zs)
            sampled_points = random.choices(
                points,
                k=num_points,
            )
            return sampled_points
    polygon_indices = random.choices(
        population=range(len(mesh.data.polygons)),
        weights=[polygon.area for polygon in mesh.data.polygons],
        k=len(xs) * len(ys) * len(zs) * 5,
    )
    surface_points = []
    for polygon_index in polygon_indices:
        verts = [
            mesh.matrix_world @ mesh.data.vertices[vertex_index].co
            for vertex_index in mesh.data.polygons[polygon_index].vertices
        ]
        # choose a random point on the surface of the polygon
        lines = [
            [verts[0], verts[1]],
            [verts[1], verts[2]],
            [verts[2], verts[0]],
        ]
        chosen_lines = random.sample(lines, k=2)
        # get the random point on each line
        points_on_line = [
            vert1 + random.uniform(0, 1) * (vert2 - vert1)
            for vert1, vert2 in chosen_lines
        ]
        # get a random point between the two points
        point = points_on_line[0] + random.uniform(0, 1) * (
            points_on_line[1] - points_on_line[0]
        )
        surface_points.append([point.x, point.y, point.z])
    # NOTE: put each of the points in a voxel
    points_per_voxel = defaultdict(list)
    for point in surface_points:
        x, y, z = point
        x_i = int((x - x_min) / x_voxel_size)
        y_i = int((y - y_min) / y_voxel_size)
        z_i = int((z - z_min) / z_voxel_size)
        points_per_voxel[(x_i, y_i, z_i)].append(point)
    # NOTE: get the rotation matrix for rotating about the x axis by 90 degrees
    rotate_x_axis_matrix = rotation_matrix(np.array([1, 0, 0]), math.pi / 2)
    # NOTE: for each voxel, choose the point that is nearest to the center of it
    vis_points = []
    for (x_i, y_i, z_i), points in points_per_voxel.items():
        # center of the voxel
        x = (x_i * x_voxel_size) + (x_voxel_size / 2) + x_min
        y = (y_i * y_voxel_size) + (y_voxel_size / 2) + y_min
        z = (z_i * z_voxel_size) + (z_voxel_size / 2) + z_min
        # find the point closest to the center of the voxel
        center = np.array([x, y, z])
        closest_point = min(points, key=lambda p: np.linalg.norm(p - center))
        if add_unity_rotation_correction:
            closest_point = rotate_x_axis_matrix @ closest_point
        vis_point = dict(x=closest_point[0], y=closest_point[1], z=closest_point[2])
        # rotate the vis_point 90 degrees around the x axis
        vis_points.append(vis_point)
    if visualize:
        # create a new parent object for the visibility points
        vis_points_parent = bpy.data.objects.new("vis_points", None)
        bpy.context.scene.collection.objects.link(vis_points_parent)
        # for each of the visibility points, create a new sphere at the point location
        for vis_point in vis_points:
            vis_point_obj = bpy.data.objects.new("vis_point", None)
            vis_point_obj.location = (vis_point["x"], vis_point["y"], vis_point["z"])
            vis_point_obj.empty_display_type = "SPHERE"
            vis_point_obj.empty_display_size = voxel_size / 2
            vis_point_obj.parent = vis_points_parent
            bpy.context.scene.collection.objects.link(vis_point_obj)
    return vis_points


def mirror_object(obj: bpy.types.Object):
    obj.scale[0] = -1
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # flip normals
    if obj.type == "MESH":
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode="OBJECT")


def rotate_for_unity(obj: bpy.types.Object, export: bool = True):
    if export == True:
        rotation_sign = -1
    elif export == False:
        rotation_sign = 1
    obj.rotation_euler[0] = math.radians(90 * rotation_sign)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def bake_image(
    image: bpy.data.images,
    TEXTURE_PATH: str,
    engine: str,
) -> None:
    bpy.context.scene.render.engine = engine
    bpy.context.scene.cycles.bake_type = "DIFFUSE"
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, save_mode="EXTERNAL")
    image.save_render(filepath=TEXTURE_PATH)


# GPT generated so doubt it works
def to_dict(
    asset_name: str,
    visibility_points: Optional[Dict[str, float]] = None,
    albedo_path: str = None,
    metallic_smoothness_path: str = None,
    normal_path: str = None,
    emission_path: str = None,
    receptacle: bool = False,
):
    obj = bpy.context.object
    obj.data.calc_normals_split()

    # Make sure it's a mesh object
    if obj.type == "MESH":
        # Get the mesh data
        mesh = obj.data

        mesh_data = {"vertices": [], "normals": [], "uvs": [], "faces": []}

        # Get the first UV layer if it exists
        uv_layer = mesh.uv_layers.active.data if mesh.uv_layers else None

        # Loop over the loops
        for loop in mesh.loops:
            # Append the vertex position to the vertices list
            vert = mesh.vertices[loop.vertex_index].co[:]
            mesh_data["vertices"].append(dict(x=vert[0], y=vert[1], z=vert[2]))

            normal = loop.normal[:]
            # Append the loop normal to the normals list
            mesh_data["normals"].append(dict(x=normal[0], y=normal[1], z=normal[2]))

            # Append the UV coordinates to the uvs list if there is a UV layer
            if uv_layer:
                uv = uv_layer[loop.index].uv[:]
                mesh_data["uvs"].append(dict(x=uv[0], y=uv[1]))

        # Loop over the polygons (faces)
        for poly in mesh.polygons:
            # Append the loop indices of the face to the faces list
            mesh_data["faces"].extend(poly.loop_indices)

        logger.debug(
            f"Ended with"
            f" {len(mesh_data['vertices'])} vertices,"
            f" {len(mesh_data['normals'])} normals,"
            f" {len(mesh_data['uvs'])} uvs,"
            f" {len(mesh_data['faces'])} faces, and "
            f" and {len(set(mesh_data['faces']))} unique elements ref'ed in faces"
        )

        return {
            # "action": "CreateRuntimeAsset",
            "name": asset_name,
            "receptacleCandidate": receptacle,
            "albedoTexturePath": albedo_path,
            "metallicSmoothnessTexturePath": metallic_smoothness_path,
            "normalTexturePath": normal_path,
            "emissionTexturePath": emission_path,
            "vertices": mesh_data["vertices"],
            "triangles": mesh_data["faces"],
            "normals": mesh_data["normals"],
            "visibilityPoints": visibility_points,
            "uvs": mesh_data["uvs"],
        }


def save_json(
    save_path: str,
    asset_name: str,
    visibility_points: Optional[Dict[str, float]] = None,
    albedo_path: str = None,
    metallic_smoothness_path: str = None,
    normal_path: str = None,
    emission_path: str = None,
    receptacle: bool = False,
) -> None:
    with open(save_path, "w") as f:
        json.dump(
            to_dict(
                asset_name=asset_name,
                visibility_points=visibility_points,
                albedo_path=albedo_path,
                metallic_smoothness_path=metallic_smoothness_path,
                normal_path=normal_path,
                emission_path=emission_path,
                receptacle=receptacle,
            ),
            f,
            indent=2,
        )


def compress_file(input_file_path: str, output_file_path: str, compresslevel=2):
    with open(input_file_path, "rb") as f_in:
        with gzip.open(output_file_path, "wb", compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(input_file_path)


def save_pickle_gzip(
    asset_name: str,
    save_path: str,
    visibility_points: Optional[List[Dict[str, float]]] = None,
    albedo_path: str = None,
    metallic_smoothness_path: str = None,
    normal_path: str = None,
    emission_path: str = None,
    receptacle: bool = False,
) -> None:
    tmp_path = save_path.replace(".gz", "")
    with open(tmp_path, "wb") as f:
        pickle.dump(
            obj=to_dict(
                asset_name=asset_name,
                visibility_points=visibility_points,
                albedo_path=albedo_path,
                metallic_smoothness_path=metallic_smoothness_path,
                normal_path=normal_path,
                emission_path=emission_path,
                receptacle=receptacle,
            ),
            file=f,
            protocol=4,
        )
    compress_file(tmp_path, save_path)


def set_base_to_origin(obj):
    # Move origin to geometry
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
    # Offset mesh so origin rests at base
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.transform.translate(value=(0, 0, obj.dimensions.z / 2))
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.transform_apply()


def find_surface_area(obj):
    # Ensure the object is a mesh
    if obj.type != "MESH":
        raise ValueError("Object must be a mesh")
    # Create a new bmesh object
    bm = bmesh.new()
    # Load the object mesh into bmesh
    bm.from_mesh(obj.data)
    # Ensure the bmesh is in a valid state
    bm.faces.ensure_lookup_table()
    # Calculate the surface area
    surface_area = sum(f.calc_area() for f in bm.faces)
    # Cleanup the bmesh
    bm.free()
    return surface_area


def is_element_garbage(obj: bpy.types.Object) -> bool:
    # Check if object is a flat planar object

    # A small number to handle floating point inaccuracies
    eps = 1e-4

    # Get the coordinates of first vertex
    first_vertex = obj.data.vertices[0].co

    # Initialize flags
    same_x, same_y, same_z = True, True, True

    # Compare all other vertices with the first one
    for v in obj.data.vertices[1:]:
        if abs(v.co.x - first_vertex.x) > eps:
            same_x = False
        if abs(v.co.y - first_vertex.y) > eps:
            same_y = False
        if abs(v.co.z - first_vertex.z) > eps:
            same_z = False
        # Early exit if none of the flags are True
        if not (same_x or same_y or same_z):
            break

    if same_x == True or same_y == True or same_z == True:
        return True
    else:
        return False


def is_object_closed(obj):
    # Start a bmesh instance
    bm = bmesh.new()

    # Load the object's data into the bmesh
    bm.from_mesh(obj.data)

    # Update the mesh with the new data
    bm.edges.ensure_lookup_table()

    # Check for any edge where 'is_boundary' property is True
    for e in bm.edges:
        if e.is_boundary:
            # Clean up the bmesh and return False
            bm.free()
            return False

    # Clean up the bmesh and return True
    bm.free()
    return True


def delete_transparent_faces_and_materials(obj: bpy.ops.object, alphaThreshold: float = 1):
    transparent_material_indices = []

    # Index all transparent materials
    for index, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    alpha_input = node.inputs['Alpha']
                    alpha = alpha_input.default_value
                    # Check if the alpha input is linked or alpha is not equal to 1 (fully opaque)
                    if alpha_input.is_linked or alpha < alphaThreshold:
                        transparent_material_indices.append(index)
                        break

    # Select all faces that use transparent materials
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    mesh = bmesh.from_edit_mesh(obj.data)
    mesh.faces.ensure_lookup_table()

    # Select faces based on material index
    for face in mesh.faces:
        if face.material_index in transparent_material_indices:
            face.select = True

    # Update mesh to reflect face selection
    bmesh.update_edit_mesh(obj.data)

    # Delete selected faces
    bpy.ops.mesh.delete(type='FACE')

    # Switch back to Object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Reverse the list to delete from the end to avoid reindexing issues
    transparent_material_indices.reverse()

    for index in transparent_material_indices:
        obj.active_material_index = index
        bpy.ops.object.material_slot_remove()

    # Remove materials that are no longer used by any object (covers entire scene)
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def weld_vertices(obj: bpy.ops.object, vertex_selection: tuple = ("all"), distance_threshold: float = 0.001):
    # Get the active object
    bpy.context.view_layer.objects.active = obj

    # Make sure the active object is a mesh
    if obj.type == "MESH":
        # Get into edit mode (necessary to perform operations on the mesh)
        bpy.ops.object.mode_set(mode="EDIT")

        # Create a bmesh object and fill it with our mesh
        bm = bmesh.from_edit_mesh(obj.data)

        # Deselect all vertices to start from a clean slate
        bpy.ops.mesh.select_all(action="DESELECT")
        bm.select_flush(False)

        # Loop over the mesh edges
        for edge in bm.edges:
            # Select border vertices if specified
            if "border" in vertex_selection and len(edge.link_faces) == 1:
                edge.verts[0].select = True
                edge.verts[1].select = True
            # Select non-border vertices if specified
            elif "nonborder" in vertex_selection and len(edge.link_faces) > 1:
                edge.verts[0].select = True
                edge.verts[1].select = True

        # If all vertices are to be selected
        if "all" in vertex_selection:
            bpy.ops.mesh.select_all(action="SELECT")

        # Update the mesh to reflect the selection
        bmesh.update_edit_mesh(obj.data)

        # Weld the selected vertices that are within the distance threshold
        bmesh.ops.remove_doubles(
            bm, verts=[v for v in bm.verts if v.select], dist=distance_threshold
        )

        # Write our bmesh back to the original mesh
        bmesh.update_edit_mesh(obj.data)

        # Get back to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
    else:
        logger.debug("The active object is not a mesh")


def regularize_normals():
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def unlink_and_nodify_connections(obj, cached_connections, channel):
    cached_connections.clear
    for mat_slot_index, mat_slot in enumerate(obj.material_slots):
        if mat_slot.material and mat_slot.material.use_nodes:
            node_tree = mat_slot.material.node_tree
            for node in node_tree.nodes:
                # Find the Principled BSDF node
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_principled_node = node
                    target_input = bsdf_principled_node.inputs.get(channel)
                    if target_input.is_linked:
                        link = target_input.links[0]

                        # Cache connection info
                        cached_connections[mat_slot_index] = {
                            'from_node': link.from_node,
                            'from_socket': link.from_socket,
                            'to_node': bsdf_principled_node,
                            'to_socket': target_input
                        }

                        # Disconnect the link
                        node_tree.links.remove(link)

                    else:
                        # Create a Value node and set its value to material's target-channel's current value
                        source_node = node_tree.nodes.new(type="ShaderNodeValue")
                        source_node.outputs[0].default_value = target_input.default_value

                        # Cache connection info
                        cached_connections[mat_slot_index] = {
                            'from_node': source_node,
                            'from_socket': source_node.outputs[0],
                            'to_node': bsdf_principled_node,
                            'to_socket': target_input
                        }

                    # If channel is Metallic, set target value to 0 (so Albedo bake works)
                    if channel == "Metallic":
                        target_input.default_value = 0


def link_to_mat_output(obj, cached_connections):
    for mat_slot_index, conn_info in cached_connections.items():
        mat_slot = obj.material_slots[mat_slot_index] if mat_slot_index < len(obj.material_slots) else None
        node_tree = mat_slot.material.node_tree

        # Find input and output nodes and sockets
        from_node = conn_info['from_node']
        from_socket = conn_info['from_socket']

        to_node = next((node for node in node_tree.nodes if node.type == 'OUTPUT_MATERIAL'), None)
        to_socket = to_node.inputs['Surface']

        node_tree.links.new(from_socket, to_socket)


def relink_connections(obj, cached_connections, channel):
    for mat_slot_index, conn_info in cached_connections.items():
        mat_slot = obj.material_slots[mat_slot_index] if mat_slot_index < len(obj.material_slots) else None
        if mat_slot and mat_slot.material and mat_slot.material.use_nodes:
            node_tree = mat_slot.material.node_tree

            # Directly use the node and socket references from cached_connections for target-channel connections
            from_node = conn_info['from_node']
            from_socket = conn_info['from_socket']
            to_node = conn_info['to_node']
            to_socket = conn_info['to_socket']

            # Create the target-channel link if both sockets are valid
            if from_socket and to_socket:
                node_tree.links.new(from_socket, to_socket)

            # Directly use the node and socket references from cached_connections for target-channel connections
            from_node = next((node for node in node_tree.nodes if node.type == 'BSDF_PRINCIPLED'), None)
            from_socket = from_node.outputs['BSDF']
            to_node = next((node for node in node_tree.nodes if node.type == 'OUTPUT_MATERIAL'), None)
            to_socket = to_node.inputs['Surface']

            # Create the Material Output link if both sockets are valid
            if from_socket and to_socket:
                node_tree.links.new(from_socket, to_socket)


def combine_maps_into_RGB_A(image_RGB, image_A, invert_A: bool = True) -> bpy.types.Image:
    # Create a new compositing node tree
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create image nodes for RGB and alpha images
    image_RGB_node = tree.nodes.new('CompositorNodeImage')
    image_RGB_node.image = image_RGB

    image_A_node = tree.nodes.new('CompositorNodeImage')
    image_A_node.image = image_A

    invert_node = tree.nodes.new("CompositorNodeInvert")
    converter_node = tree.nodes.new("CompositorNodeConvertColorSpace")
    converter_node.from_color_space = 'sRGB'
    converter_node.to_color_space = 'Linear'

    # Output
    output_node = tree.nodes.new('CompositorNodeComposite')
    # output_node.use_alpha = True

    # Connect RGB image to Composite node's image input
    links.new(image_RGB_node.outputs['Image'], output_node.inputs['Image'])

    if (invert_A):
        # Connect alpha image to Invert node's color input
        links.new(image_A_node.outputs['Image'], invert_node.inputs['Color'])
    
        # Connect Invert node's color output to Composite node's alpha input
        links.new(invert_node.outputs['Color'], converter_node.inputs['Image'])
        links.new(converter_node.outputs['Image'], output_node.inputs['Alpha'])

    else:
        # Connect alpha image to Composite node's alpha input
        links.new(image_A_node.outputs['Image'], output_node.inputs['Alpha'])

    # Set render resolution
    bpy.context.scene.render.resolution_x = image_RGB.size[0]
    bpy.context.scene.render.resolution_y = image_RGB.size[1]

    # Render and return the result
    bpy.ops.render.render(write_still=True)
    return bpy.data.images['Render Result']

def delete_everything():
    materials = bpy.data.materials
    # delete all the materials
    for material in materials:
        bpy.data.materials.remove(material)
    # clear and delete everything
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def get_json_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.json")


def get_picklegz_save_path(out_dir, object_name):
    return os.path.join(out_dir, f"{object_name}.pkl.gz")


# TODO cleanup, make args match APIs better
def glb_to_thor(
    object_path: str,
    output_dir: str,
    annotations_file: str,
    save_obj: bool,
    engine="CYCLES",
    save_as_json=False,
    relative_texture_paths=True,
):
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    object_name, ext = os.path.splitext(os.path.basename(object_path))

    if annotations_file == "":
        annotation_dict = {
            "scale": 1,
            "pose_z_rot_angle": 0,
            "z_axis_scale": True,
            "ref_category": "Objaverse",
        }
    elif os.path.isdir(annotations_file):
        annotations_file = os.path.join(annotations_file, f"{object_name}.json.gz")
        with gzip.open(annotations_file, "rt") as f:
            annotation_dict = json.load(f)
    elif annotations_file:
        with (
            gzip.open(annotations_file, "rt")
            if annotations_file.endswith(".gz")
            else open(annotations_file, "r")
        ) as f:
            annotation_dict = json.load(f)

        if object_name in annotation_dict:
            annotation_dict = annotation_dict[object_name]

    # DEPENDS ON WORLD-SCALE
    # object_size = 0.023 # needs to be hooked up to UID-key value input
    # object_size_is_height = True # binary for whether object_size represents height, or longest side
    # object_canonical_rotation = 3.14 # needs to be hooked up to UID-key value input

    max_side_length_meters = annotation_dict["scale"]

    logger.debug(f"max_side_length_meters: {max_side_length_meters}")

    if "receptacle" in annotation_dict:
        receptacle = annotation_dict["receptacle"]
    else:
        receptacle = (
            annotation_dict["ref_category"] in util.get_receptacle_object_types()
        )

    # Reset scene
    reset_scene()
    purge_orphan_data()

    logger.debug("Loading model into scene and flattening hierarchy...")

    # Load new model into Blender
    load_model(object_path)
    original_object = bpy.context.selected_objects[0]

    # Flatten hierarchy
    flatten_scene_hierarchy()

    # Delete all objects that are not meshes
    delete_nonmesh_objects()
    logger.debug("Consolidating contiguous elements...")

    # # Remove any garbage elements
    # bpy.ops.object.select_all(action="SELECT")
    # for obj in bpy.context.selected_objects:
    #     if is_element_garbage(obj) == True:
    #         bpy.data.objects.remove(obj)

    # Merge all objects together
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.join()

    source_object = bpy.data.objects[0]
    bpy.context.view_layer.objects.active = source_object

    # NUCLEAR OPTION
    # Delete all transparent or semi-transparent faces
    delete_transparent_faces_and_materials(source_object, 1)

    # Run initial weld of close-contact vertices
    logger.debug("Welding close-contact vertices...")
    logger.debug("PRE-BORDER WELD: " + str(len(source_object.data.vertices)))
    weld_vertices(obj=source_object, vertex_selection=("border"), distance_threshold=0.0001)
    logger.debug("POST-BORDER WELD: " + str(len(source_object.data.vertices)))

    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.context.object.data.auto_smooth_angle = math.radians(180)

    # Check whether mesh is open or not
    open_mesh = is_mesh_open(source_object)
    logger.debug("MESH HAS BORDER: " + str(open_mesh))

    # Rotate object into canonical rotation
    logger.debug(annotation_dict["pose_z_rot_angle"])

    bpy.context.object.rotation_mode = "XYZ"
    source_object.rotation_euler[2] += annotation_dict["pose_z_rot_angle"]

    # Scale the object to its canonical size, and determine its polycount
    if annotation_dict["z_axis_scale"]:
        scale_factor = source_object.dimensions.z / max_side_length_meters
    else:
        max_side_length_meters_in_original = max(
            source_object.dimensions.x,
            source_object.dimensions.y,
            source_object.dimensions.z,
        )
        scale_factor = max_side_length_meters_in_original / max_side_length_meters

    source_object.scale /= scale_factor

    # Freeze transforms of all objects
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Move object to (0,0,0)
    source_object.location = (0, 0, 0)
    set_base_to_origin(source_object)
    bpy.ops.object.transform_apply()

    # check if target_poly_count is more than source_poly_count
    source_surface_area = find_surface_area(source_object)
    logger.debug("TOTAL SURFACE AREA: " + str(source_surface_area))
    min_local_poly_count = 100
    initial_poly_count = 5000
    additional_polys_per_square_meter = 150
    target_poly_count = (
        initial_poly_count + additional_polys_per_square_meter * source_surface_area
    )
    logger.debug("TARGET POLY-COUNT: " + str(target_poly_count))
    if source_surface_area > 8:
        texture_size = 1024
    elif source_surface_area > 1:
        texture_size = 1024
    else:
        texture_size = 1024

    logger.debug("Creating target object...")
    # Duplicate source object to create target object
    bpy.ops.object.duplicate()
    target_object = bpy.context.selected_objects[0]
    bpy.ops.object.shade_smooth()

    # Separate objects by continguous elements into long string, and remerge them (Necessary for target-bake regardless of decimation)
    # (You have to reorganize source objects as well in order to maintain the proper alignment of source and target objects)
    logger.debug(
        "Separating source and target objects by contiguous elements, stringing them out, and remerging them..."
    )

    # KEEP THIS AS SMALL AS POSSIBLE! LONG STRINGS OF SEPARATED ELEMENTS IMPACT BOTH DECIMATION AND BAKE QUALITY, IN WAYS THAT THEY LOGICALLY SHOULDN'T!
    element_spacing_amount = 1

    bpy.ops.object.select_all(action="DESELECT")
    source_object.select_set(True)
    bpy.context.view_layer.objects.active = source_object
    source_object.name = "source_object"
    bpy.ops.mesh.separate(type="LOOSE")
    source_objects = bpy.context.selected_objects

    logger.debug(f"SOURCE-OBJECTS LENGTH: {str(len(source_objects))}")
    for i in range(0, len(source_objects)):
        source_objects[i].name = "object_" + str(i).zfill(4) + "_source"
        source_objects[i].location = (
            element_spacing_amount * i,
            element_spacing_amount * i,
            element_spacing_amount * i,
        )
        source_objects[i].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    source_object = bpy.context.selected_objects[0]
    purge_orphan_data()

    bpy.ops.object.select_all(action="DESELECT")
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object
    target_object.name = "target_object"
    bpy.ops.mesh.separate(type="LOOSE")
    target_objects = bpy.context.selected_objects

    bpy.ops.object.select_all(action="DESELECT")
    logger.debug(f"TARGET-OBJECTS LENGTH: {str(len(target_objects))}")
    for i in range(0, len(target_objects)):
        target_objects[i].name = "object_" + str(i).zfill(4) + "_target"
        target_objects[i].location = (
            element_spacing_amount * i,
            element_spacing_amount * i,
            element_spacing_amount * i,
        )
        target_objects[i].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    target_object = bpy.context.selected_objects[0]
    purge_orphan_data()

    # DECIMATION PROCESS

    # Get polygon count
    source_poly_count = len(target_object.data.polygons)
    logger.debug(
        f"POLYGON COUNT PRE-DECIMATION: {source_poly_count} (Target is {target_poly_count})"
    )

    # Calculate whether decimating object to fulfill the calculated polygon density is necessary
    if source_poly_count > target_poly_count:
        logger.debug("Poly-count limit exceeded. Now decimating...")

        decimation_iter_current = 0
        decimation_iter_max = 1

        # ITERATIVE COLLAPSE DECIMATION
        while (
            target_poly_count + 10 < len(target_object.data.polygons)
            and decimation_iter_current < decimation_iter_max
        ):
            # Find mesh's minimum decimation threshold, to determine whether extra weld is necessary
            dec_min = get_min_decimation(target_object)
            logger.debug(f"MINIMUM DECIMATION POLY-COUNT: {str(dec_min)}")

            target_object.select_set(True)
            bpy.context.view_layer.objects.active = target_object

            # FALLBACK: Extra vertex-weld step, if necessary. Check if this mesh is a temperamental diva that won't decimate without a preemptive vertex-merge, and then add some extra buffer
            buffer_coefficient = 1.5
            if buffer_coefficient * dec_min > target_poly_count:
                logger.debug(
                    f"Additional weld required. Pre: {str(len(target_object.data.vertices))}"
                )
                weld_vertices(obj=source_object, vertex_selection=("nonborder"), distance_threshold=0.001)
                logger.debug(f"Post: {str(len(target_object.data.vertices))}")
                regularize_normals()

                # ADD STEP HERE TO REMOVE USELESS TWO-VERTEX ELEMENTS??? IT'D BE VERY TRICKY, SINCE YOU'D NEED TO REMOVE CORRESPONDING SOURCE-ONES...

                # If vertex-merge still doesn't create sufficiently decimatable mesh, then the squeaky axle gets the grease, and we decimate it as much as we can, with some buffer
                dec_min = get_min_decimation(target_object)
                if buffer_coefficient * dec_min > target_poly_count:
                    # Adding "squeaky-axle" coefficient to difficult-to-decimate asset, to strike balance between decent quality and some amount of decimation
                    squeaky_axle_coefficient = 1.5
                    target_poly_count = squeaky_axle_coefficient * dec_min
                logger.debug(f"NEW MINIMUM DECIMATION POLY-COUNT IS {str(target_poly_count)}")
            else:
                logger.debug(
                    "No additional weld necessary. Proceeding to decimation..."
                )

            decimation_ratio = target_poly_count / len(target_object.data.polygons)

            logger.debug(
                f"DECIMATION RATIO: {str(target_poly_count)} / {str(len(target_object.data.polygons))} = {str(decimation_ratio)}"
            )

            target_object.select_set(True)
            bpy.context.view_layer.objects.active = target_object

            dec_mod_name = "Decimate_" + str(decimation_iter_current).zfill(4)
            bpy.ops.object.modifier_add(type="DECIMATE")
            bpy.context.object.modifiers[-1].name = dec_mod_name
            bpy.context.object.modifiers[dec_mod_name].ratio = decimation_ratio
            bpy.context.object.modifiers[dec_mod_name].use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=dec_mod_name)
            logger.debug(
                f"POST-DECIMATION POLY-COUNT: {str(len(target_object.data.polygons))}"
            )

            decimation_iter_current += 1
    else:
        logger.debug("Poly-count limit subceeded. Skipping decimation...")

    purge_orphan_data()

    # MATERIAL CREATION

    # Create UV map
    bpy.ops.object.select_all(action="DESELECT")

    logger.debug("Creating new UV layout...")
    create_uv_map(target_object, texture_size)

    # Set up new material for target object
    bake_mat = bpy.data.materials.new(name=object_name + "_mat")
    bake_mat.use_nodes = True
    bake_mat_bsdf = bake_mat.node_tree.nodes["Principled BSDF"]

    # Albedo map setup
    bake_mat_ti_albedo = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    bake_mat_ti_albedo.image = bpy.data.images.new(
        "Target_Object_Albedo_Bake", texture_size, texture_size
    )
    # bpy.data.images["Target_Object_Albedo_Bake"].source = 'FILE'
    albedo_texture = bpy.data.images.new(
        str(object_name) + "_albedo", width=texture_size, height=texture_size
    )
    bake_mat.node_tree.links.new(
        bake_mat_ti_albedo.outputs["Color"], bake_mat_bsdf.inputs["Base Color"]
    )

    # Specular reset
    bake_mat_bsdf.inputs["Specular"].default_value = 0

    # Normal map setup
    bake_mat_ti_normal = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    bake_mat_ti_normal.image = bpy.data.images.new(
        "Target_Object_Normal_Bake", texture_size, texture_size
    )
    # bpy.data.images["Target_Object_Albedo_Bake"].source = 'FILE'
    bake_mat_ti_normal.image.colorspace_settings.name = "Non-Color"

    bake_mat_nm = bake_mat.node_tree.nodes.new(type="ShaderNodeNormalMap")
    bake_mat.node_tree.links.new(
        bake_mat_ti_normal.outputs["Color"], bake_mat_nm.inputs["Color"]
    )
    bake_mat.node_tree.links.new(
        bake_mat_nm.outputs["Normal"], bake_mat_bsdf.inputs["Normal"]
    )

    # # Metallic map setup
    # bake_mat_ti_metallic = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    # bake_mat_ti_metallic.image = bpy.data.images.new(
    #     "Target_Object_Metallic_Bake", texture_size, texture_size
    # )

    # bake_mat.node_tree.links.new(
    #     bake_mat_ti_metallic.outputs["Color"], bake_mat_bsdf.inputs["Metallic"]
    # )

    # # Roughness map setup
    # bake_mat_ti_roughness = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    # bake_mat_ti_roughness.image = bpy.data.images.new(
    #     "Target_Object_Roughness_Bake", texture_size, texture_size
    # )

    # bake_mat.node_tree.links.new(
    #     bake_mat_ti_roughness.outputs["Color"], bake_mat_bsdf.inputs["Roughness"]
    # )

    # Emission map setup
    bake_mat_ti_emission = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    bake_mat_ti_emission.image = bpy.data.images.new(
        "Target_Object_Emission_Bake", texture_size, texture_size
    )
    emission_texture = bpy.data.images.new(
        str(object_name) + "_emission", width=texture_size, height=texture_size
    )
    bake_mat.node_tree.links.new(
        bake_mat_ti_emission.outputs["Color"], bake_mat_bsdf.inputs["Emission"]
    )

    # # Transparency map setup
    # bake_mat_ti_transparency = bake_mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    # bake_mat_ti_transparency.image = bpy.data.images.new(
    #     "Target_Object_Transparency_Bake", texture_size, texture_size
    # )
    # transparency_texture = bpy.data.images.new(
    #     str(object_name) + "_transparency", width=texture_size, height=texture_size
    # )
    # bake_mat.node_tree.links.new(
    #     bake_mat_ti_transparency.outputs["Color"], bake_mat_bsdf.inputs["Alpha"]
    # )

    # Apply new material to target object
    bpy.context.view_layer.objects.active = target_object
    for i in range(0, len(target_object.material_slots)):
        bpy.ops.object.material_slot_remove()
    bpy.ops.object.material_slot_add()
    target_object.data.materials[0] = bake_mat

    logger.debug(
        "Source and target objects strung out, remerged, and target object decimated. Running bake from source to target..."
    )
    logger.debug(f"ATLAS TEXTURE SIZE: {str(texture_size)} x {str(texture_size)}")
    # Set up baking parameters
    bpy.ops.object.select_all(action="DESELECT")
    source_object.select_set(True)
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object

    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = (
        0.01 * annotation_dict["scale"] + 0.01
    )
    logger.debug("CAGE EXTRUSION: " + str(0.02 * annotation_dict["scale"] + 0.01))
    bpy.context.scene.render.bake.margin = texture_size

    # Create modular channel two-node and two-slot array...
    socket_connections = {}

    # Execute source-to-target object bakes
    bpy.ops.object.select_all(action="DESELECT")
    source_object.select_set(True)
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object

    # Albedo bake

    # This must be done now because the Metallic channels need to be zeroed out for the Albedo bake to work properly
    if len(source_object.material_slots) > 0:
        unlink_and_nodify_connections(source_object, socket_connections, "Metallic")

    bake_mat.node_tree.nodes.active = bake_mat_ti_albedo
    bpy.ops.object.bake(type="DIFFUSE")

    albedo_map_name = "albedo.png"
    # Save out albedo map texture
    data_block = bpy.data.images["Target_Object_Albedo_Bake"]
    logger.debug(f"Saving {albedo_map_name}...")
    albedo_save_path = os.path.join(output_dir, albedo_map_name)
    data_block.save_render(filepath=albedo_save_path)

    # Normal bake
    bake_mat.node_tree.nodes.active = bake_mat_ti_normal
    bpy.ops.object.bake(type="NORMAL")

    normal_map_name = "normal.png"
    # Save out normal map texture
    data_block = bpy.data.images["Target_Object_Normal_Bake"]
    logger.debug(f"Saving {normal_map_name}...")
    normal_save_path = os.path.join(output_dir, normal_map_name)
    data_block.save_render(filepath=normal_save_path)

    # Metallic bake
    link_to_mat_output(source_object, socket_connections)

    bake_mat.node_tree.nodes.active = bake_mat_ti_metallic
    bpy.ops.object.bake(type="EMIT")

    metallic_map_name = "metallic.png"
    # Save out roughness map texture
    data_block = bpy.data.images["Target_Object_Metallic_Bake"]
    logger.debug(f"Saving {metallic_map_name}...")
    metallic_save_path = os.path.join(output_dir, metallic_map_name)
    data_block.save_render(filepath=metallic_save_path)

    relink_connections(source_object, socket_connections, "Metallic")

    # Roughness bake
    bake_mat.node_tree.nodes.active = bake_mat_ti_roughness
    bpy.ops.object.bake(type="ROUGHNESS")

    roughness_map_name = "roughness.png"
    # Save out roughness map texture
    data_block = bpy.data.images["Target_Object_Roughness_Bake"]
    logger.debug(f"Saving {roughness_map_name}...")
    roughness_save_path = os.path.join(output_dir, roughness_map_name)
    data_block.save_render(filepath=roughness_save_path)

    # Composite metallic and roughness maps into metallic-smoothness map
    metallic_smoothness_map_name = "metallic_smoothness.png"
    # Save out metallic_smoothness map texture
    data_block = combine_maps_into_RGB_A(bpy.data.images["Target_Object_Metallic_Bake"], bpy.data.images["Target_Object_Roughness_Bake"], True)
    logger.debug(f"Saving {metallic_smoothness_map_name}...")
    metallic_smoothness_save_path = os.path.join(output_dir, metallic_smoothness_map_name)
    data_block.save_render(filepath=metallic_smoothness_save_path)

    # Emission bake
    bake_mat.node_tree.nodes.active = bake_mat_ti_emission
    bpy.ops.object.bake(type="EMIT")

    emission_map_name = "emission.png"
    # Save out emission map texture
    data_block = bpy.data.images["Target_Object_Emission_Bake"]
    logger.debug(f"Saving {emission_map_name}...")
    emission_save_path = os.path.join(output_dir, emission_map_name)
    data_block.save_render(filepath=emission_save_path)

    # # Transparency bake
    # if len(source_object.material_slots) > 0:
    #     unlink_and_nodify_connections(source_object, socket_connections, "Alpha")

    # link_to_mat_output(source_object, socket_connections)
    # bake_mat.node_tree.nodes.active = bake_mat_ti_transparency
    # bpy.ops.object.bake(type="EMIT")

    # transparency_map_name = "transparency.png"
    # # Save out transparency map texture
    # data_block = bpy.data.images["Target_Object_Transparency_Bake"]
    # logger.debug(f"Saving {transparency_map_name}...")
    # transparency_save_path = os.path.join(output_dir, transparency_map_name)
    # data_block.save_render(filepath=transparency_save_path)

    # relink_connections(source_object, socket_connections, "Alpha")

    albedo_path = (
        albedo_map_name
        if relative_texture_paths
        else os.path.join(output_dir, f"{albedo_map_name}")
    )
    normal_path = (
        normal_map_name
        if relative_texture_paths
        else os.path.join(output_dir, f"{normal_map_name}")
    )
    metallic_smoothness_path = (
        metallic_smoothness_map_name
        if relative_texture_paths
        else os.path.join(output_dir, f"{metallic_smoothness_map_name}")
    )
    metallic_path = (
        metallic_map_name
        if relative_texture_paths
        else os.path.join(output_dir, f"{metallic_map_name}")
    )
    roughness_path = (
        roughness_map_name
        if relative_texture_paths
        else os.path.join(output_dir, f"{roughness_map_name}")
    )
    emission_path = (
        emission_map_name
        if relative_texture_paths
        else os.path.join(output_dir, f"{emission_map_name}")
    )
    # transparency_path = (
    #     transparency_map_name
    #     if relative_texture_paths
    #     else os.path.join(output_dir, f"{transparency_map_name}")
    # )

    # save_path = os.path.join(output_dir, f"{object_name}.json")
    json_save_path = get_json_save_path(output_dir, object_name)
    picklegz_save_path = get_picklegz_save_path(output_dir, object_name)

    # De-stringing decimated and baked target object (and source object, for reference)
    bpy.ops.object.select_all(action="DESELECT")
    source_object.select_set(True)
    bpy.context.view_layer.objects.active = source_object
    bpy.ops.mesh.separate(type="LOOSE")
    source_objects = bpy.context.selected_objects

    bpy.ops.object.select_all(action="DESELECT")
    for i in range(0, len(source_objects)):
        source_objects[i].name = "object_" + str(i).zfill(4) + "_source"
        source_objects[i].location = (
            element_spacing_amount * -i,
            element_spacing_amount * -i,
            element_spacing_amount * -i,
        )
        source_objects[i].select_set(True)

    # Recombine all source objects into single one
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    source_object = bpy.context.selected_objects[0]
    source_object.name = "source_object"
    purge_orphan_data()

    bpy.ops.object.select_all(action="DESELECT")
    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object
    bpy.ops.mesh.separate(type="LOOSE")
    target_objects = bpy.context.selected_objects

    # When repositioning target object elements, make any open ones double-sided by duplicating their geometry and flipping the normals
    for i in range(0, len(target_objects)):
        bpy.ops.object.select_all(action="DESELECT")
        target_objects[i].name = "object_" + str(i).zfill(4) + "_target"
        target_objects[i].location = (
            element_spacing_amount * -i,
            element_spacing_amount * -i,
            element_spacing_amount * -i,
        )
        target_objects[i].select_set(True)
        if is_mesh_open(target_objects[i]):
            logger.debug("MESH-ELEMENT HAS BORDER: TRUE")
            bpy.ops.object.duplicate()
            bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.flip_normals()
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            logger.debug("MESH-ELEMENT HAS BORDER: FALSE")

    # Recombine all target objects into single one
    bpy.ops.object.select_all(action="SELECT")
    source_object.select_set(False)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

    bpy.ops.object.join()
    target_object = bpy.context.selected_objects[0]
    target_object.name = object_name
    purge_orphan_data()

    # Delete source object (Comment when iterating (to use as reference comparison), uncomment for final rollout)
    # bpy.data.objects.remove(source_object)

    target_object.select_set(True)
    bpy.context.view_layer.objects.active = target_object

    # Ensure that edges will be soft in Unity Engine
    polygons = target_object.data.polygons
    polygons.foreach_set("use_smooth", [True] * len(polygons))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.mark_sharp(clear=True)
    # Correct normals (Not necessary)
    # bpy.ops.mesh.normals_make_consistent(inside=False)
    # bpy.ops.object.mode_set(mode="OBJECT")
    # Select all faces (Necessary for correct face normal direction, for some reason)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Correct for json export, which causes a rotation of -90 degrees around the x-axis in Unity and a x-axis flip
    mirror_object(target_object)

    if save_obj:
        obj_filename = os.path.join(output_dir, f"{object_name}.obj")
        blend_file_path = bpy.data.filepath
        obj_materials = False
        bpy.ops.export_scene.obj(
            filepath=obj_filename,
            check_existing=True,
            axis_forward="-Z",
            axis_up="Y",
            filter_glob="*.obj;*.mtl",
            use_selection=False,
            use_animation=False,
            use_mesh_modifiers=True,
            use_edges=True,
            use_smooth_groups=False,
            use_smooth_groups_bitflags=False,
            use_normals=True,
            use_uvs=True,
            use_materials=obj_materials,
            use_triangles=True,
            use_nurbs=False,
            use_vertex_groups=False,
            use_blen_objects=True,
            group_by_object=False,
            group_by_material=False,
            keep_vertex_order=False,
            global_scale=1,
            path_mode="AUTO",
        )

    rotate_for_unity(target_object, export=True)

    visibility_points = get_visibility_points(target_object, visualize=True)

    if not save_as_json:
        save_pickle_gzip(
            asset_name=object_name,
            save_path=picklegz_save_path,
            visibility_points=visibility_points,
            albedo_path=albedo_path,
            metallic_smoothness_path=metallic_smoothness_path,
            normal_path=normal_path,
            emission_path=emission_path,
            receptacle=receptacle,
        )
    else:
        save_json(
            asset_name=object_name,
            save_path=json_save_path,
            visibility_points=visibility_points,
            albedo_path=albedo_path,
            metallic_smoothness_path=metallic_smoothness_path,
            normal_path=normal_path,
            emission_path=emission_path,
            receptacle=receptacle,
        )

    # Re-orient object post-export, for visual feedback
    mirror_object(target_object)
    rotate_for_unity(target_object, export=False)
    bpy.ops.object.select_all(action="DESELECT")

    bpy.data.objects["vis_points"].select_set(True)
    vispoints = bpy.context.selected_objects[0]
    mirror_object(vispoints)
    rotate_for_unity(vispoints, export=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default="",
        help="Path to annotations file for object metadata, if this path is a directory,"
        " will assume annotations can be found at os.path.basename(object_path.replace('.glb', '.json.gz')).",
    )

    parser.add_argument(
        "--receptacle", action="store_true", help="Whether the object is a receptacle."
    )

    parser.add_argument(
        "--relative_texture_paths",
        action="store_true",
        help="Save textures as relative paths.",
    )

    parser.add_argument("--obj", action="store_true")

    parser.add_argument("--save_as_json", action="store_true")

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    glb_to_thor(
        object_path=args.object_path,
        output_dir=args.output_dir,
        engine=args.engine,
        annotations_file=args.annotations,
        save_obj=args.obj,
        save_as_json=args.save_as_json,
        relative_texture_paths=args.relative_texture_paths,
    )
