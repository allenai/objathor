import os
from pathlib import Path

ABS_PATH_OF_OBJATHOR = os.path.abspath(os.path.dirname(Path(__file__)))
THOR_COMMIT_ID = "40679c517859e09c1f2a5e39b65ee7f33fcfdd48"

OBJATHOR_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".objathor-assets")

VISION_LLM = "gpt-4-vision-preview"
TEXT_LLM = "gpt-4-1106-preview"
