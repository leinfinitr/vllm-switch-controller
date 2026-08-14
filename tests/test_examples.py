from pathlib import Path

from yaml import safe_load

from vllm_switch_controller.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIGS = {
    "launcher": REPOSITORY_ROOT / "configs" / "models.launcher.example.yaml",
    "external": REPOSITORY_ROOT / "configs" / "models.external.example.yaml",
}


def test_example_config_names_are_semantic_and_legacy_names_are_absent():
    assert set(path.name for path in (REPOSITORY_ROOT / "configs").glob("*.example.yaml")) == {
        path.name for path in EXAMPLE_CONFIGS.values()
    }
    assert not (REPOSITORY_ROOT / "configs" / "models.example.yaml").exists()
    assert not (REPOSITORY_ROOT / "configs" / "models.example2.yaml").exists()


def test_example_configs_validate():
    for path in EXAMPLE_CONFIGS.values():
        config = load_config(path)
        assert config.models


def test_launcher_example_owns_process_commands_and_external_example_does_not():
    launcher = safe_load(EXAMPLE_CONFIGS["launcher"].read_text(encoding="utf-8"))
    external = safe_load(EXAMPLE_CONFIGS["external"].read_text(encoding="utf-8"))

    assert all(model["launch_command"] for model in launcher["models"].values())
    assert all("cwd" in model and "env" in model for model in launcher["models"].values())
    assert all(
        "launch_command" not in model and "cwd" not in model and "env" not in model
        for model in external["models"].values()
    )
