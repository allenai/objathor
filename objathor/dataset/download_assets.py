import copy

from objathor.dataset import DEFAULT_DSC, load_assets_path
from objathor.dataset.download_annotations import download_parse_args

if __name__ == "__main__":
    args = download_parse_args(description="Download assets from the dataset")

    dsc = copy.deepcopy(DEFAULT_DSC)
    dsc.VERSION = args.version
    dsc.PATH = args.path

    path = load_assets_path(dsc)
    print(f"Assets downloaded to {path}")
