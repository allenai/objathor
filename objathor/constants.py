import os
from pathlib import Path

ABS_PATH_OF_OBJATHOR = os.path.abspath(os.path.dirname(Path(__file__)))
# THOR_COMMIT_ID = "b92f8068d993d8242fb920808a2814cdb5f7ed6e"
THOR_COMMIT_ID = "da6d276430305ecee222e1de16171223518a7dc3"  # latest and greatest

OBJATHOR_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".objathor-assets")

# VISION_LLM = "gpt-4o-2024-08-06"
# TEXT_LLM = "gpt-4o-2024-08-06"
VISION_LLM = "gpt-4.1-mini"
TEXT_LLM = "gpt-4.1-mini"
