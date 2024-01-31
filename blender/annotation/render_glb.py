import os
import math
import sys
import argparse

from mathutils import Vector
import bpy
from typing import Sequence

# blender -b -P /path/to/your/render_script.py -- /path/to/your/model.glb /path/to/output/directory 0,90,180


def render_glb(glb_path: str, output_dir: str, angles: Sequence[float]):
    # Set render resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Check if there are any other specific objects (e.g., default cube) and remove them
    # Replace 'Cube' with the name of the object you want to remove
    cube = bpy.data.objects.get("Cube")
    if cube:
        bpy.data.objects.remove(cube, do_unlink=True)

    # Load GLB file
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # Set output directory for rendered images
    os.makedirs(output_dir, exist_ok=True)

    # Define list of azimuth angles (in degrees) for rendering
    azimuths = angles

    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not mesh_objects:
        print("No mesh objects found in the scene.")
        exit(-1)

    # Merge all mesh objects into a single object
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.select_all(action="DESELECT")

    for obj in mesh_objects:
        obj.select_set(True)

    bpy.ops.object.join()

    # Get the merged object
    merged_object = bpy.context.active_object

    # Center the merged object at the scene's origin
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="BOUNDS")

    # Move the object to the scene's origin
    merged_object.matrix_world.translation = (0, 0, 0)

    bpy.context.view_layer.update()

    # Select the imported object
    obj = bpy.context.active_object

    # Normalize object size
    max_size = max(obj.dimensions)
    obj.scale /= max_size

    # Set up rendering parameters
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"

    scene.render.film_transparent = True  # Enable transparent background

    # Calculate camera position for each azimuth
    for azimuth in azimuths:
        # Calculate camera position
        radians = math.radians(azimuth)
        x = math.cos(radians)
        y = math.sin(radians)

        dist = 1.7
        # Set camera location
        camera_location = (dist * x, dist * y, 0.8)

        # Set camera orientation
        camera = bpy.data.objects["Camera"]

        camera.location = camera_location

        # Point the camera towards the origin
        camera.rotation_mode = "XYZ"
        look_at = Vector((0, 0, 0))  # Origin coordinates
        direction = look_at - Vector(camera.location)
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()

        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # bpy.ops.view3d.camera_to_view_selected()

        # Replace the default light with a new light source
        for _obj in bpy.context.scene.objects:
            if _obj.type == "LIGHT":
                bpy.data.lights.remove(_obj.data)  # Remove the default light

        # Create a new light source (e.g., Point light)
        light_data = bpy.data.lights.new(name="NewLight", type="POINT")
        light_object = bpy.data.objects.new(name="NewLight", object_data=light_data)
        scene.collection.objects.link(light_object)

        # Position the light slightly to the right from the camera's viewpoint
        camera = bpy.data.objects["Camera"]
        light_object.location = (
            camera.location.x + 0.0,
            camera.location.y + 0.0,
            camera.location.z + 1.0,
        )  # Adjust position

        # Set light energy and other properties as needed
        light_object.data.energy = 300.0  # Adjust the light intensity

        # Access the Compositor
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links

        # Clear default nodes
        for node in nodes:
            nodes.remove(node)

        # Create new nodes
        render_layers_node = nodes.new("CompositorNodeRLayers")
        alpha_over_node = nodes.new("CompositorNodeAlphaOver")

        # Set up node connections
        links.new(render_layers_node.outputs["Image"], alpha_over_node.inputs[1])
        links.new(render_layers_node.outputs["Alpha"], alpha_over_node.inputs[2])

        # Create a transparent background
        bg_node = nodes.new("CompositorNodeOutputFile")
        bg_node.format.file_format = "PNG"
        bg_node.base_path = output_dir  # Set the output folder path
        bg_node.file_slots.new("Alpha")
        bg_node.file_slots[0].use_node_format = False

        # Render and save the final result with only the object's pixels and transparent background
        bpy.context.scene.render.filepath = os.path.join(
            output_dir, f"render_{azimuth}.png"
        )
        bpy.ops.render.render(write_still=True)

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glb_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where render images will be writen.",
    )

    parser.add_argument(
        "--angles",
        type=str,
        default="0,90,180,270",
        help="Comma separated azimuth angle (degrees) list to render the object around.",
    )

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    angles = [float(angle) for angle in args.angles.split(",")]
    render_glb(glb_path=args.glb_path, output_dir=args.output_dir, angles=angles)
