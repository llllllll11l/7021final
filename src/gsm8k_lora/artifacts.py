"""Download the experiment directory hosted on Hugging Face."""

import argparse
from pathlib import Path
import shutil

from .common import DEFAULT_CONFIG, read_json

PROJECT_ROOT = DEFAULT_CONFIG.parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "results" / "seed13" / "artifacts.json"


def copy_snapshot(source, destination):
    """Copy the downloaded directory into an empty local artifact directory."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Choose an empty destination: {destination}")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    count = sum(path.is_file() for path in destination.rglob("*"))
    print(f"Downloaded {count} experiment files to {destination}")


def fetch(record, destination):
    from huggingface_hub import snapshot_download

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Choose an empty destination: {destination}")
    snapshot = snapshot_download(record["repo_id"], repo_type="dataset", revision=record["revision"],
                                 allow_patterns=[record["directory"] + "/**"])
    copy_snapshot(Path(snapshot) / record["directory"], destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="Download the published seed-13 directory")
    download.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    download.add_argument("--destination", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()
    fetch(read_json(args.source), args.destination)


if __name__ == "__main__":
    main()
