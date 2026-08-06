from pathlib import Path

from yaml import safe_load

from controller.schemas import (
    CPU_BACKUP_CAPABILITIES,
    CPU_BACKUP_PROTOCOL_VERSION,
    CPU_BACKUP_REQUIRED_CAPABILITIES,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "compatibility" / "v0.1.yaml"
EXPECTED_COMPONENTS = {
    "controller": {
        "tag": "v0.1.5",
        "commit": "8b64ad232c8eba5d0da265abd93bd7b061db0549",
    },
    "engine": {
        "tag": "aipc2-v0.1.0",
        "commit": "71071ce4d0bc65e38acf2da76eb8c6fb05b9454d",
    },
    "benchmark": {
        "tag": "v0.1.8",
        "commit": "e4e388acc33977bee7ca19d72a2959fc736d76ab",
    },
}


def manifest() -> dict:
    return safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_published_suite_manifest_binds_tag_and_commit_for_each_component():
    data = manifest()

    assert data["schema_version"] == 1
    assert data["status"] == "published"
    for component, expected in EXPECTED_COMPONENTS.items():
        actual = data["components"][component]
        assert {key: actual[key] for key in expected} == expected
        assert len(actual["commit"]) == 40


def test_suite_manifest_matches_controller_protocol_constants():
    cpu_backup = manifest()["protocols"]["cpu_backup"]

    assert cpu_backup["version"] == CPU_BACKUP_PROTOCOL_VERSION
    assert set(cpu_backup["required_capabilities"]) == CPU_BACKUP_REQUIRED_CAPABILITIES
    assert set(cpu_backup["optional_capabilities"]) == (
        CPU_BACKUP_CAPABILITIES - CPU_BACKUP_REQUIRED_CAPABILITIES
    )


def test_long_term_compatibility_doc_links_manifest_and_excludes_collection_commits():
    documentation = (REPOSITORY_ROOT / "docs" / "compatibility.md").read_text(encoding="utf-8")

    assert "../compatibility/v0.1.yaml" in documentation
    assert "1b3919d8c210af05f6ea8b29fff33fb8d07e6c1d" not in documentation
    assert "36e08b7e6393e7c9ab9747ca7b3c95562353c998" not in documentation
    assert "9ad35876ba1b7921f8e1547698a1a8412709078e" not in documentation
