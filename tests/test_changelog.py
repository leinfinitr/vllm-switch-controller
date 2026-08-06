import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
RELEASES = {
    "0.1.0": "2026-08-04",
    "0.1.1": "2026-08-05",
    "0.1.2": "2026-08-05",
    "0.1.3": "2026-08-05",
    "0.1.4": "2026-08-05",
    "0.1.5": "2026-08-05",
}


def test_changelog_covers_every_published_controller_tag():
    headings = dict(re.findall(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$", CHANGELOG, re.M))

    assert headings == RELEASES


def test_changelog_compare_links_cover_adjacent_release_tags():
    releases = list(RELEASES)
    for previous, current in zip(releases, releases[1:], strict=False):
        expected = (
            f"[{current}]: https://github.com/leinfinitr/vllm-switch/compare/"
            f"v{previous}...v{current}"
        )
        assert expected in CHANGELOG

    assert (
        "[Unreleased]: https://github.com/leinfinitr/vllm-switch/compare/v0.1.5...HEAD" in CHANGELOG
    )
