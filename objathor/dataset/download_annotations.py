import copy
from argparse import ArgumentParser

from objathor.dataset import DEFAULT_DSC, load_annotations_path


def download_parse_args(description: str):
    parser = ArgumentParser(
        description=description,
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_DSC.VERSION,
        help=f"The version to download (default: {DEFAULT_DSC.VERSION}).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_DSC.PATH,
        help=f"The dir to download to (default: {DEFAULT_DSC.PATH}).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = download_parse_args(
        description="Download annotations from the dataset repository.",
    )

    dsc = copy.deepcopy(DEFAULT_DSC)
    dsc.VERSION = args.version
    dsc.PATH = args.path

    path = load_annotations_path(dsc)
    print(f"Annotations downloaded to {path}")
