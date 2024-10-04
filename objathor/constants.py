import os
from pathlib import Path

ABS_PATH_OF_OBJATHOR = os.path.abspath(os.path.dirname(Path(__file__)))
THOR_COMMIT_ID = "b92f8068d993d8242fb920808a2814cdb5f7ed6e"

OBJATHOR_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".objathor-assets")

VISION_LLM = "gpt-4o-2024-08-06"
TEXT_LLM = "gpt-4o-2024-08-06"
