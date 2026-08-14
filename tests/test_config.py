import pytest
from pydantic import ValidationError

from vllm_switch_controller.config import ControllerConfig, load_config


def test_load_config_parses_models_and_controller(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  a:
    backend_url: http://127.0.0.1:8101
    served_model_name: model-a
controller:
  port: 9000
  startup_awake_model: a
  metrics_path: results/events.jsonl
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, ControllerConfig)
    assert config.controller.port == 9000
    assert config.controller.startup_awake_model == "a"
    assert config.models["a"].backend_url == "http://127.0.0.1:8101"
    assert config.models["a"].sleep_level == 1


def test_config_rejects_unknown_startup_model(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  a:
    backend_url: http://127.0.0.1:8101
    served_model_name: model-a
controller:
  startup_awake_model: missing
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="startup_awake_model"):
        load_config(config_path)


def test_model_env_does_not_inject_deprecated_disk_backup_variable():
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {
                    "backend_url": "http://a",
                    "served_model_name": "a",
                }
            }
        }
    )

    assert "VLLM_EXACT_DISK_BACKUP_DIR" not in config.models["a"].env


def test_model_env_keeps_canonical_exact_disk_override():
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {
                    "backend_url": "http://a",
                    "served_model_name": "a",
                    "env": {"VLLM_EXACT_DISK_BACKUP_DIR": "/mnt/backup-a"},
                }
            }
        }
    )

    assert config.models["a"].env["VLLM_EXACT_DISK_BACKUP_DIR"] == "/mnt/backup-a"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "models": {
                    "a": {
                        "backend_url": "http://127.0.0.1:8101",
                        "served_model_name": "a",
                    }
                },
                "unexpected": True,
            },
            "Extra inputs are not permitted",
        ),
        (
            {
                "models": {
                    "a": {
                        "backend_url": "http://127.0.0.1:8101",
                        "served_model_name": "a",
                        "typo": True,
                    }
                }
            },
            "Extra inputs are not permitted",
        ),
        (
            {
                "models": {
                    "a": {
                        "backend_url": "http://127.0.0.1:8101",
                        "served_model_name": "a",
                    }
                },
                "controller": {"port": 0},
            },
            "greater than or equal to 1",
        ),
        (
            {
                "models": {
                    "a": {
                        "backend_url": "http://127.0.0.1:8101",
                        "served_model_name": "a",
                    }
                },
                "controller": {"request_timeout_s": 0},
            },
            "greater than 0",
        ),
    ],
)
def test_config_rejects_extra_fields_and_invalid_runtime_constraints(payload, message):
    with pytest.raises(ValidationError, match=message):
        ControllerConfig.model_validate(payload)


def test_controller_defaults_to_loopback():
    config = ControllerConfig.model_validate(
        {
            "models": {
                "a": {
                    "backend_url": "http://127.0.0.1:8101",
                    "served_model_name": "a",
                }
            }
        }
    )

    assert config.controller.host == "127.0.0.1"


@pytest.mark.parametrize("wake_tags", [[], ["weights", "weights"], [""]])
def test_config_rejects_ambiguous_wake_tags(wake_tags):
    with pytest.raises(ValidationError):
        ControllerConfig.model_validate(
            {
                "models": {
                    "a": {
                        "backend_url": "http://127.0.0.1:8101",
                        "served_model_name": "a",
                        "wake_tags": wake_tags,
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "controller",
    [
        {"cpu_memory_reclaim_available_ratio": 0.10},
        {"cpu_memory_recovery_available_ratio": 0.20},
        {"cpu_memory_recovery_available_bytes": 1024},
    ],
)
def test_config_rejects_incomplete_memory_pressure_pairs(controller):
    with pytest.raises(ValidationError):
        ControllerConfig.model_validate(
            {
                "models": {
                    "a": {
                        "backend_url": "http://a",
                        "served_model_name": "a",
                    }
                },
                "controller": controller,
            }
        )
