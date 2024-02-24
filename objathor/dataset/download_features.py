from objathor.dataset import load_features_dir, DatasetSaveConfig
from objathor.dataset.download_annotations import download_parse_args

if __name__ == "__main__":
    args = download_parse_args(description="Download features from the dataset")

    dsc = DatasetSaveConfig(
        VERSION=args.version,
        BASE_PATH=args.path,
    )

    path = load_features_dir(dsc)
    print(f"Features downloaded to {path}")
