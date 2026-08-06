import tomllib
from importlib.metadata import version as distribution_version
from pathlib import Path

from packaging.version import Version
from yaml import safe_load

import controller

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION_NAME = "vllm-switch-controller"
EXPECTED_DEVELOPMENT_VERSION = "0.2.0.dev0"
LATEST_RELEASE_VERSION = "0.1.5"
LATEST_RELEASE_DATE = "2026-08-05"
LATEST_RELEASE_COMMIT = "8b64ad232c8eba5d0da265abd93bd7b061db0549"


def project_metadata() -> dict:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]


def citation_metadata() -> dict:
    return safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def workflow_text(name: str) -> str:
    return (REPOSITORY_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_release_workflow_rejects_tag_version_mismatches():
    workflow = workflow_text("release.yml")

    assert "RELEASE_TAG: ${{ github.ref_name }}" in workflow
    assert 'expected_tag = f"v{project_version}"' in workflow
    assert "actual_tag != expected_tag" in workflow


def test_development_version_is_pep440_and_consistent():
    versions = {
        project_metadata()["version"],
        controller.__version__,
        distribution_version(DISTRIBUTION_NAME),
    }

    assert versions == {EXPECTED_DEVELOPMENT_VERSION}
    assert Version(EXPECTED_DEVELOPMENT_VERSION).is_devrelease
    assert Version(EXPECTED_DEVELOPMENT_VERSION) > Version(LATEST_RELEASE_VERSION)


def test_citation_identifies_latest_release_instead_of_unreleased_checkout():
    citation = citation_metadata()

    assert citation["version"] == LATEST_RELEASE_VERSION
    assert citation["date-released"].isoformat() == LATEST_RELEASE_DATE
    assert citation["commit"] == LATEST_RELEASE_COMMIT
    assert citation["authors"] == [{"name": "vLLM Switch contributors"}]
    assert "latest published release" in citation["message"]
    assert "development checkout" in citation["message"]


def test_packaging_workflows_run_metadata_consistency_gates_before_build():
    gates = (
        "uv run python -m pytest tests/test_version.py -q",
        "uv run python -m pytest tests/test_compatibility_manifest.py -q",
    )

    for workflow_name in ("ci.yml", "release.yml"):
        workflow = workflow_text(workflow_name)
        for gate in gates:
            assert gate in workflow
            assert workflow.index(gate) < workflow.index("uv build")
