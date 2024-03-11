# ObjaTHOR

Objaverse asset annotator and importer for use in THOR.

## Installation

Install ai2thor:

```bash
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+40679c517859e09c1f2a5e39b65ee7f33fcfdd48
```

Install other dependencies:

```bash
pip install objathor[annotation]
```

Here the following extras are installed: `annotation` to use openai to generate annotations. Also for annotation functionality you must install `nltk` [Install nltk](#nltk-dependencies). To generate renders and convert 'glb' models in the conversion pipeline you must [Install Blender](#blender-install-instructions) .

From source:

```bash
pip install -e ".[annotation]"
```

We recommend setting an environment variable with your OpenAI key:

```bash
export OPENAI_API_KEY=[Your key]
```

If we're planning to annotate objects for which we don't have pre-generated
views in S3 (see an example under [Annotation](#annotation) below), we also need to install blender either as an extra (as shown above) or as an application:

### Blender install instructions
Installing  the `Blender` as a module:
```bash
pip install -e bpy"
```
Installing  the Blender as a module, requires a python `3.10` environment.

Or installing blender as an application:
[Blender install instructions](https://docs.blender.org/manual/en/latest/getting_started/installing/index.html)

If application is not in the cannonical directories you may need to pass `blender_installation_path` to scripts that use Blender.


### NLTK dependencies

Install `nltk` on this commit by running:

```bash
pip install git+https://github.com/nltk/nltk@582e6e35f0e6c984b44ec49dcb8846d9c011d0a8
```

During the first run, NLTK dependencies are automatically installed, but we can also install them ahead:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet2022'); nltk.download('brown'); nltk.download('averaged_perceptron_tagger')"
```


### Pre-generated synset definition embeddings for Annotation

For automatic annotation to assign likely synsets given the automatically generated asset description, we can
pre-install pre-generated embeddings for all synset definitions (this can be useful if we cannot write into our home
directory at run time):

```bash
mkdir ~/.objathor_data
curl https://prior-datasets.s3.us-east-2.amazonaws.com/vida-synset-embeddings/synset_definition_embeddings_single.pkl.gz -o ~/.objathor_data/synset_definition_embeddings_single.pkl.gz
```

### AI2-THOR binary pre-downloading

Assuming we're running on a remote Linux server, we can pre-download the THOR binaries with:

```bash
python -c "from ai2thor.controller import Controller; from objathor.constants import THOR_COMMIT_ID; c=Controller(download_only=True, platform='CloudRendering', commit_id=THOR_COMMIT_ID)"
```

(`platform='OSXIntel64'` would be used for a MacOS environment).

## Usage

### Annotation

You must install the `annotation` extra requirement through pip, ad have blender installed,
either standalone or as a module. The following command will generate annotation, via GPT-4,
and also generate the conversion to a valid THOR asset.

```bash
OUTPUT_DIR=/path/to/output
python -m objathor.main \
--uid 0070ac4bf50b496387365843d4bf5432 \
--output "$OUTPUT_DIR"
```

### GLB to THOR asset conversion

From the repository root run:

```
python 
-m objathor.asset_conversion.pipeline_to_thor 
--object_ids=000074a334c541878360457c672b6c2e 
--output_dir=<some_absolute_path>
--extension=.msgpack.gz
 --annotations=<annotations_file_path> 
--live 
--blender_as_module
```

Where `object_ids` is a string of comma separated list of `Objaverse` object ids to process.
`output_dir` is an absolute path indicating where to write the output of the conversion.
`annotations` is optional, and is the path to an annotations file as generated by the process described above.

Run `python -m objathor.asset_conversion.pipeline_to_thor --help` for other options.
