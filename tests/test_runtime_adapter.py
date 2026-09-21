import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime import FakeBackend, allocation

from switch_runtime.adapters.vllm import VllmProvider, VllmSleepBackend, config_from_environment
from switch_runtime.config import RuntimeConfig
from switch_runtime.runtime import BackupRuntime, ResidencyState


@pytest.fixture
def runtime(tmp_path):
    value = BackupRuntime(FakeBackend(), RuntimeConfig(disk_root=tmp_path, chunk_bytes=4096))
    yield value
    value.close()


@pytest.fixture
def provider(runtime):
    backend = VllmSleepBackend.__new__(VllmSleepBackend)
    backend.runtime = runtime
    backend.checkpoint_pending = False
    backend.initialized = True
    backend.needs_prebackup = False
    value = VllmProvider()
    value.backend = lambda: backend
    return value


def test_fixed_weights_rejects_mutation_without_importing_engine(provider):
    with pytest.raises(RuntimeError, match="fixed weights"):
        with provider.worker_context(None, "mutation"):
            pytest.fail("mutation reached the engine")


def test_l2_pending_blocks_inference_and_snapshot_publication(provider, runtime):
    device, _ = allocation(runtime)
    with provider.worker_context(None, "sleep", kwargs={"level": 2}):
        runtime.sleep(())
    with provider.worker_context(None, "wake"):
        runtime.wake_up()
    assert device.data[0] == 0
    with pytest.raises(RuntimeError, match="reconstruction"):
        with provider.worker_context(None, "inference"):
            pass
    with pytest.raises(RuntimeError, match="reconstruction"):
        provider.worker_event(None, "prepare_cpu")


def test_failed_reconstruction_poisoned_even_when_mapping_is_awake(provider, runtime):
    allocation(runtime)
    provider.backend().checkpoint_pending = True
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        with provider.worker_context(None, "reload"):
            raise RuntimeError("checkpoint failure")
    assert runtime.residency_state == ResidencyState.RECOVERY_REQUIRED


def test_direct_sleep_prepare_failure_leaves_mappings_awake(provider, runtime):
    allocation(runtime)
    runtime.backend.fail_copy = True
    with pytest.raises(RuntimeError, match="copy failed"):
        with provider.worker_context(None, "sleep"):
            runtime.sleep("weights")
    assert runtime.residency_state == ResidencyState.AWAKE


@pytest.mark.parametrize("option", ["eplb", "transfer", "lora"])
def test_unsupported_configuration_fails_before_model_load(option):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=True),
        parallel_config=SimpleNamespace(enable_eplb=option == "eplb"),
        weight_transfer_config=object() if option == "transfer" else None,
        lora_config=object() if option == "lora" else None,
    )
    with pytest.raises(ValueError):
        VllmProvider().validate_config(config)


def test_legacy_env_aliases_are_parsed_outside_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_EXACT_DISK_BACKUP_ENABLED", "true")
    monkeypatch.setenv("VLLM_EXACT_DISK_BACKUP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXACT_DISK_BACKUP_DIRECT_IO", "false")
    config = config_from_environment()
    assert config.disk_enabled
    assert config.disk_root == tmp_path
    assert not config.direct_io


def test_mixed_parameter_buffer_allocation_is_not_frozen(provider, runtime):
    from switch_runtime.contracts import SavePolicy

    immutable, first = allocation(runtime)
    mixed, second = allocation(runtime)

    def tensor(region):
        storage = SimpleNamespace(data_ptr=lambda: region.address, nbytes=lambda: region.size_bytes)
        return SimpleNamespace(device=SimpleNamespace(type="cuda"), untyped_storage=lambda: storage)

    model = SimpleNamespace(
        parameters=lambda: [tensor(first), tensor(second)],
        buffers=lambda: [tensor(second)],
    )
    worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))
    provider.backend().freeze(worker)
    assert runtime.allocations[first.address].region.policy == SavePolicy.IMMUTABLE
    assert runtime.allocations[second.address].region.policy == SavePolicy.SNAPSHOT
    runtime.prepare_cpu_backup("weights")
    mixed.data[0] = 99
    runtime.sleep("weights")
    runtime.wake_up()
    assert immutable.data[0] == 37 and mixed.data[0] == 99


def test_stats_identify_actual_runtime_without_mutating_storage(provider, runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__file__="/engine/vllm/__init__.py"))
    device, _ = allocation(runtime)
    before = runtime.get_cpu_backup_pool_stats()
    stats = provider.worker_event(None, "stats")
    identity = stats["runtime_identity"]
    assert identity["sleep_backend"] == "switch"
    assert identity["provider_api_version"] == 1
    assert identity["engine_module_path"] == "/engine/vllm/__init__.py"
    assert Path(identity["runtime_module_path"]).name == "runtime.py"
    assert identity["package_version"]
    assert identity["pid"] == stats["pid"]
    assert identity["runtime_config"]["disk_enabled"] is False
    assert runtime.get_cpu_backup_pool_stats() == before
    assert device.data[0] == 37
