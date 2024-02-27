import os
from pathlib import Path

ABS_PATH_OF_OBJATHOR = os.path.abspath(os.path.dirname(Path(__file__)))
THOR_COMMIT_ID = "2d0f5e678d1d9fef5f8a25990cc3051699d12f97"

OBJATHOR_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".objathor-assets")

VISION_LLM = "gpt-4-vision-preview"
TEXT_LLM = "gpt-4-1106-preview"
