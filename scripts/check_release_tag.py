"""Fail closed when a release tag does not match package metadata."""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def project_version(root: Path = REPOSITORY_ROOT) -> str:
    """Return the PEP 440 version declared by the release checkout."""

    with (root / "pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def validate_release_tag(tag: str, root: Path = REPOSITORY_ROOT) -> None:
    """Raise when *tag* is not the exact ``v<project-version>`` tag."""

    version = project_version(root)
    expected = f"v{version}"
    if tag != expected:
        raise ValueError(f"release tag {tag!r} does not match package version {version!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Require the release tag to equal v<project-version>."
    )
    parser.add_argument(
        "tag",
        nargs="?",
        default=os.environ.get("RELEASE_TAG"),
        help="Tag to validate (defaults to RELEASE_TAG).",
    )
    args = parser.parse_args(argv)
    if not args.tag:
        parser.error("a tag argument or RELEASE_TAG is required")
    try:
        validate_release_tag(args.tag)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
