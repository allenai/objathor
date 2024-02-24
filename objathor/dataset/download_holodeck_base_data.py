from objathor.dataset import DatasetSaveConfig, load_holodeck_base
from objathor.dataset.download_annotations import download_parse_args

if __name__ == "__main__":
    args = download_parse_args(description="Download files needed to run holodeck.")

    args.version = "2023_09_23"
    dsc = DatasetSaveConfig(
        VERSION=args.version,
        BASE_PATH=args.path,
    )

    path = load_holodeck_base(dsc)
    print(f"Holodeck base downloaded to {path}")
