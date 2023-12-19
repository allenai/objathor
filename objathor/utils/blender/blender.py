from typing import Sequence, List, Optional
import os
import subprocess
import glob

import objaverse

from objathor.constants import ABS_PATH_OF_OBJATHOR
from objathor.objaverse.util import get_blender_installation_path


def render_glb_from_angles(
    glb_uid: str,
    base_dir: str = "data",
    angles: Sequence[float] = (0, 90, 180, 270),
    timeout: Optional[int] = 2 * 60,
) -> Optional[List[str]]:
    glb_path = objaverse.load_objects([glb_uid])[glb_uid]
    object_out_dir = os.path.join(base_dir, glb_uid)

    command = (
        f"{get_blender_installation_path()}"
        f" --background"
        f" --python {os.path.join(ABS_PATH_OF_OBJATHOR, 'utils', 'blender', 'render_glb.py')}"
        f" --"
        f' "{os.path.abspath(glb_path)}"'
        f' "{os.path.abspath(object_out_dir)}"'
        + " "
        + " ".join([str(angle) for angle in angles])
    )

    print(f"For {glb_uid}, running command: {command}")

    out = ""
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
        print(f"---- Command ran successfully for {glb_uid} at path {glb_path}")
        return glob.glob(os.path.join(os.path.abspath(object_out_dir), "*.png"))

    return None


if __name__ == "__main__":

    def main():
        render_glb_from_angles("0070ac4bf50b496387365843d4bf5432")

    main()
