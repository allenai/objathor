# Dataset

This README file contains some basic information about how to create, upload, and use an objathor asset dataset (i.e. optimized assets and their annotations).

# Downloading and using generated datasets

Objathor datasets (i.e. assets, annotations, and (if available) retrieval features) are versioned by their date.
You can download the assets,  annotations, and metadata separately for a specific version of the dataset, for example:
```bash
python -m objathor.dataset.download_annotations --version 2023_07_28
python -m objathor.dataset.download_assets --version 2023_07_28
```
By default, these will save to `~/.objathor-assets/2023_07_28/annotations.json.gz` and `~/.objathor-assets/2023_07_28/assets/` respectively.
You can change this the directory by using the `--path` argument.

In some cases (i.e. if the [generate_holodeck_features.py](generate_holodeck_features.py) script has been run), assets
will also have features associated with them (e.g. CLIP features of the rendered objects). These are needed to be able
to use these objects with [Holodeck](https://github.com/allenai/Holodeck). Note that Holodeck uses a `2023_09_23` 
version of the assets, given this version you can download the associated features using:
```bash
python -m objathor.dataset.download_features --version 2023_09_23
```

Finally, if you're planning to use Holodeck, you'll also want to run
```bash
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
```
which collects other metadata/annotations/features neeed when generating Holodeck houses (this data is
saved to `~/.objathor-assets/holodeck/2023_09_23/`).

# Preparing and uploading a dataset

## Preparing the annotations file

The annotations file is a JSON file that contains the annotations for all the objects in the dataset.

```bash
python -m objathor.dataset.prepare_annotations \
--save-dir DIRECTORY_OF_OPTIMIZED_ASSETS \
--output-file /PATH/TO/YOUR_DATASET_UID/annotations.json.gz
```

## Preparing the asset tarball
We exclude some files and directories that are not necessary for the dataset (this includes the per-object annotations which are uploaded separately as a single big file).
```bash
cd /path/to/your/optimized/assets
tar \
--exclude='annotations.json.gz' \
--exclude='thor_metadata.json' \
--exclude='.DS_Store' \
--exclude='._*' \
--exclude='.Spotlight-V100' \
--exclude='.Trashes' \
--exclude='.fseventsd' \
--exclude='.VolumeIcon.icns' \
--exclude='.apdisk' \
--exclude='.TemporaryItems' \
--exclude='*.lock' \
-cvf /PATH/TO/YOUR_DATASET_UID/assets.tar DIRECTORY_OF_OPTIMIZED_ASSETS
```

## Uploading to CloudFlare R2

Download and set up `rclone` from [here](https://rclone.org/downloads/) (also available via brew). You can then run:
```bash
rclone copy -P /PATH/TO/YOUR_DATASET_UID/assets.tar r2:your-bucket/YOUR_DATASET_UID/
```
and similarly for other files.

For a large asset.tar file, the above command might be quite slow, you can potentially speed it up by changing a few flags:
```bash
rclone copy -P \
--s3-chunk-size 50M \
--transfers 300 \
--s3-upload-concurrency 300 \
--s3-chunk-size 50M \
--ignore-checksum \
--s3-disable-checksum \
/PATH/TO/YOUR_DATASET_UID/assets.tar r2:your-bucket/YOUR_DATASET_UID/
```
