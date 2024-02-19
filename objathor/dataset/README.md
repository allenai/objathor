# Dataset

This readme file contains some basic information about how to create, upload, and use an objathor asset dataset (i.e. optimized assets and their annotations).


## Create a dataset folder to store the assets and annotations
```bash
mkdir /PATH/TO/YOUR_DATASET_UID
```

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
tar \
--exclude='annotations.json.gz' \
--exclude='.DS_Store' \
--exclude='._*' \
--exclude='.Spotlight-V100' \
--exclude='.Trashes' \
--exclude='.fseventsd' \
--exclude='.VolumeIcon.icns' \
--exclude='.apdisk' \
--exclude='.TemporaryItems'\
-cvf /PATH/TO/YOUR_DATASET_UID/assets.tar DIRECTORY_OF_OPTIMIZED_ASSETS
```

## Uploading to CloudFlare R2

Download and set up `rclone` from [here](https://rclone.org/downloads/) (also available via brew).

```bash
rclone copy /PATH/TO/YOUR_DATASET_UID/assets.tar r2:your-bucket/YOUR_DATASET_UID/
```
