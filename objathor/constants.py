import os
from pathlib import Path

ABS_PATH_OF_OBJATHOR = os.path.abspath(os.path.dirname(Path(__file__)))
THOR_COMMIT_ID = "f6701d695de3cb10708fc0686e814276db242ecc"

OBJATHOR_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".objathor-assets")

VISION_LLM = "gpt-4-vision-preview"
TEXT_LLM = "gpt-4-1106-preview"
