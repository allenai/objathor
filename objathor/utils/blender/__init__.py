import glob
import os
import subprocess
import traceback
from typing import Sequence, List, Optional

from objathor.asset_conversion.util import (
    get_blender_installation_path,
    compress_image_to_ssim_threshold,
)
from objathor.constants import ABS_PATH_OF_OBJATHOR


class BlenderRenderError(Exception):
    pass


def render_glb_from_angles(
    glb_path: str,
    save_dir: str,
    angles: Sequence[float] = (0, 90, 180, 270),
    timeout: Optional[int] = 2 * 60,
    save_as_jpg: bool = True,
    verbose: bool = False,
    blender_as_module: Optional[bool] = None,
    overwrite: bool = False,
) -> List[str]:
    save_dir = os.path.abspath(save_dir)
    glb_path = os.path.abspath(glb_path)

    os.makedirs(save_dir, exist_ok=True)

    if not overwrite:
        expected_extension = ".jpg" if save_as_jpg else ".png"
        expected_paths = [
            os.path.join(os.path.join(save_dir, f"{float(a):0.1f}{expected_extension}"))
            for a in angles
        ]
        if all(os.path.exists(p) for p in expected_paths):
            return expected_paths

    if blender_as_module is None:
        try:
            import bpy

            blender_as_module = True
        except ImportError:
            blender_as_module = False

    if not blender_as_module:
        command = (
            f"{get_blender_installation_path()}"
            f" --background"
            f" --python {os.path.join(ABS_PATH_OF_OBJATHOR, 'utils', 'blender', 'render_glb.py')}"
            f" --"
            f' --glb_path="{glb_path}"'
            f' --output_dir="{save_dir}"'
            f' --angles={",".join([str(angle) for angle in angles])}'
        )
    else:
        command = (
            f"python"
            f" -m"
            f" objathor.utils.blender.render_glb"
            f" --"
            f' --glb_path="{glb_path}"'
            f' --output_dir="{save_dir}"'
            f' --angles={",".join([str(angle) for angle in angles])}'
        )

    if verbose:
        print(f"For {os.path.basename(glb_path)}, running command: {command}")

    process = None
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        out, _ = process.communicate(timeout=timeout)
        out = out.decode()
        result_code = process.returncode
        if result_code != 0:
            raise subprocess.CalledProcessError(result_code, command)
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
            process.wait(timeout=timeout)
        result_code = -1
        out = f"Blender render command timed out, command: {command}"
    except subprocess.CalledProcessError as e:
        result_code = e.returncode
        out = traceback.format_exc()

    if verbose:
        print(out)
        print(f"Rendering: exited with code {result_code}")

    success = result_code == 0

    if success:
        if verbose:
            print(f"Blender renders successfully generated for {glb_path}")
        blender_render_paths = glob.glob(os.path.join(save_dir, "*.png"))
        if save_as_jpg:
            for brp in blender_render_paths:
                compress_image_to_ssim_threshold(
                    input_path=brp,
                    output_path=brp[:-4] + ".jpg",
                    threshold=0.99,
                )
                os.remove(brp)
            return glob.glob(os.path.join(save_dir, "*.jpg"))
        else:
            return blender_render_paths

    raise BlenderRenderError(
        f"Blender render failed for {glb_path}. Command: {command}. Output: {out}"
    )
