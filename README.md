# ObjaTHOR

Objaverse asset annotator and importer for use in THOR. 

## Installation

```bash
pip install -r requirements.txt
```

We recommend setting an environment variable with your OpenAI key:

```bash
export OPENAI_API_KEY=[Your key]
```

## Usage

To generate the initial annotation for a uid in Objaverse for which we have pre-rendered views:

```bash
OUTPUT_DIR = /path/to/output
python -m objathor.main \
--uid 0070ac4bf50b496387365843d4bf5432 \
--output "$OUTPUT_DIR"/0070ac4bf50b496387365843d4bf5432.json.gz
```

If we don't have pre-rendered views, just add `--local_render`:

```bash
OUTPUT_DIR=/path/to/output
python -m objathor.main \
--uid 0070ac4bf50b496387365843d4bf5432 \
--output "$OUTPUT_DIR"/0070ac4bf50b496387365843d4bf5432.json.gz \
--local_render
```
