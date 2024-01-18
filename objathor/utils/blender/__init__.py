import glob
import os
import subprocess
from typing import Sequence, List, Optional

from objathor.asset_conversion.util import get_blender_installation_path
from objathor.constants import ABS_PATH_OF_OBJATHOR


def render_glb_from_angles(
    glb_path: str,
    save_dir: str,
    angles: Sequence[float] = (0, 90, 180, 270),
    timeout: Optional[int] = 2 * 60,
) -> Optional[List[str]]:
    command = (
        f"{get_blender_installation_path()}"
        f" --background"
        f" --python {os.path.join(ABS_PATH_OF_OBJATHOR, 'utils', 'blender', 'render_glb.py')}"
        f" --"
        f' "{os.path.abspath(glb_path)}"'
        f' "{os.path.abspath(save_dir)}"'
        + " "
        + " ".join([str(angle) for angle in angles])
    )

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
        out = f"Command timed out, command: {command}"
    except subprocess.CalledProcessError as e:
        result_code = e.returncode
        print(f"Blender call error: {e.output}")
        out = e.output

    print(out)

    print(f"Exited with code {result_code}")

    success = result_code == 0

    if success:
        print(f"---- Command ran successfully for {glb_path}")
        return glob.glob(os.path.join(os.path.abspath(save_dir), "*.png"))

    return None
