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

If we're planning to annotate objects for which we don't have pre-generated
views in S3 (see an example under [Annotation](#annotation) below), we also need to install blender:

[Blender install instructions](https://docs.blender.org/manual/en/latest/getting_started/installing/index.html)

## Usage

### Annotation

To generate the initial annotation for a uid in Objaverse for which we have pre-rendered views in S3, like

[https://objaverse-im.s3.us-west-2.amazonaws.com/0070ac4bf50b496387365843d4bf5432/009.png](https://objaverse-im.s3.us-west-2.amazonaws.com/0070ac4bf50b496387365843d4bf5432/009.png),

we can just:

```bash
OUTPUT_DIR = /path/to/output
python -m objathor.main \
--uid 0070ac4bf50b496387365843d4bf5432 \
--output "$OUTPUT_DIR"/0070ac4bf50b496387365843d4bf5432.json.gz
```

If we don't have pre-rendered views, we can just add `--local_render`:

```bash
OUTPUT_DIR=/path/to/output
python -m objathor.main \
--uid 0070ac4bf50b496387365843d4bf5432 \
--output "$OUTPUT_DIR"/0070ac4bf50b496387365843d4bf5432.json.gz \
--local_render
```
