from objathor.dataset import load_assets_path, DatasetSaveConfig
from objathor.dataset.download_annotations import download_parse_args

if __name__ == "__main__":
    args = download_parse_args(description="Download assets from the dataset")

    dsc = DatasetSaveConfig(
        VERSION=args.version,
        BASE_PATH=args.path,
    )
    path = load_assets_path(dsc)
    print(f"Assets downloaded to {path}")
