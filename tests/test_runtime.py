import ctypes
import dataclasses
import threading
import uuid

import pytest

from switch_runtime.config import RuntimeConfig
from switch_runtime.contracts import MemoryRegion, SavePolicy
from switch_runtime.runtime import BackupRuntime, CpuBackupState, ResidencyState


class Buffer:
    def __init__(self, size):
        self.data = bytearray(size)

    @property
    def address(self):
        return ctypes.addressof(ctypes.c_char.from_buffer(self.data))

    @property
    def size_bytes(self):
        return len(self.data)

    def view(self):
        return memoryview(self.data)


class FakeStream:
    def submit(self, slot, source, destination):
        ctypes.memmove(
            destination, ctypes.addressof(ctypes.c_char.from_buffer(source)), len(source)
        )

    def wait(self, slot):
        return 0.001

    def synchronize(self):
        pass

    def close(self):
        pass


class FakeBackend:
    pin_memory = False

    def __init__(self):
        self.copies = []
        self.maps = []
        self.unmaps = []
        self.host_flushes = 0
        self.fail_copy = False
        self.fail_unmap = False

    def allocate_host(self, size_bytes):
        return Buffer(size_bytes)

    def copy(self, destination, source, size_bytes):
        if self.fail_copy:
            raise RuntimeError("copy failed")
        self.copies.append((destination, source, size_bytes))
        ctypes.memmove(destination, source, size_bytes)

    def map(self, region):
        self.maps.append(region.address)

    def unmap(self, region):
        if self.fail_unmap:
            raise RuntimeError("unmap failed")
        self.unmaps.append(region.address)
        ctypes.memset(region.address, 0, region.size_bytes)

    def synchronize(self):
        pass

    def empty_device_cache(self):
        pass

    def empty_host_cache(self):
        self.host_flushes += 1

    def restore_stream(self, slots):
        return FakeStream()


@pytest.fixture
def runtime(tmp_path):
    value = BackupRuntime(FakeBackend(), RuntimeConfig(disk_root=tmp_path, chunk_bytes=4096))
    yield value
    value.close()


def allocation(runtime, size=4096, tag="weights", policy=SavePolicy.IMMUTABLE):
    buffer = Buffer(size)
    buffer.data[:] = bytes([37]) * size
    region = MemoryRegion(uuid.uuid4().hex, buffer.address, size, 0, tag, buffer, policy)
    runtime.register(region)
    return buffer, region


def test_immutable_reuse_and_actual_restore(runtime):
    device, region = allocation(runtime)
    expected = bytes(device.data)
    first = runtime.prepare_cpu_backup("weights")
    assert first["prepared_bytes"] == region.size_bytes
    for _ in range(3):
        prepared = runtime.prepare_sleep("weights")
        assert prepared["prepared_bytes"] == 0
        assert prepared["reused_bytes"] == region.size_bytes
        runtime.sleep("weights", skip_prepare=True)
        assert bytes(device.data) != expected
        runtime.wake_up()
        assert bytes(device.data) == expected
    assert len(runtime.backend.copies) == 4


def test_mutable_buffer_is_refreshed_each_sleep(runtime):
    device, _ = allocation(runtime, policy=SavePolicy.SNAPSHOT)
    runtime.prepare_cpu_backup("weights")
    device.data[:] = bytes([92]) * len(device.data)
    expected = bytes(device.data)
    runtime.sleep("weights")
    runtime.wake_up()
    assert bytes(device.data) == expected


def test_prepare_lease_blocks_reclaim_and_abort_releases(runtime):
    device, region = allocation(runtime)
    runtime.prepare_sleep("weights")
    assert not runtime.reclaim(region.size_bytes)["released"]
    runtime.abort_sleep_prepare("weights")
    runtime._poll_and_release_cpu_backups()
    assert runtime.cpu_backup_pool.reserved_bytes == 0
    assert device.data[0] == 37
    assert runtime.backend.host_flushes == 1


def test_prepared_transaction_cannot_be_replaced_or_freed(runtime):
    device, region = allocation(runtime, policy=SavePolicy.SNAPSHOT)
    runtime.prepare_sleep("weights")
    with pytest.raises(RuntimeError, match="sleep_prepared"):
        runtime.prepare_cpu_backup("weights")
    with pytest.raises(RuntimeError, match="prepared sleep"):
        runtime.unregister(region.address)
    assert not runtime.reclaim(region.size_bytes)["released"]
    runtime.sleep("weights", skip_prepare=True)
    runtime.wake_up()
    assert device.data[0] == 37


def test_prepare_failure_never_unmaps_and_can_retry(runtime):
    devices = [allocation(runtime), allocation(runtime)]
    runtime.backend.fail_copy = True
    with pytest.raises(RuntimeError, match="copy failed"):
        runtime.prepare_sleep("weights")
    assert not runtime.backend.unmaps
    assert runtime.residency_state == ResidencyState.AWAKE
    assert all(
        data.cpu_backup_state == CpuBackupState.INVALID
        for data in runtime.allocations.values()
        if data.cpu_backup_buffer is not None
    )
    runtime.backend.fail_copy = False
    runtime.sleep("weights")
    runtime.wake_up()
    assert all(device.data[0] == 37 for device, _ in devices)


def test_partial_unmap_fails_closed(runtime):
    allocation(runtime)
    runtime.backend.fail_unmap = True
    with pytest.raises(RuntimeError, match="unmap failed"):
        runtime.sleep("weights")
    assert runtime.residency_state == ResidencyState.RECOVERY_REQUIRED
    with pytest.raises(RuntimeError):
        runtime.wake_up()


def test_reallocated_pointer_gets_new_identity_and_no_snapshot(runtime):
    device, region = allocation(runtime)
    runtime.prepare_cpu_backup("weights")
    runtime.unregister(region.address)
    device.data[:] = bytes([93]) * len(device.data)
    new = dataclasses.replace(region, region_id=uuid.uuid4().hex)
    runtime.register(new)
    runtime.sleep("weights")
    runtime.wake_up()
    assert device.data[0] == 93


def test_partial_wake_and_checkpoint_reset(runtime):
    weights, _ = allocation(runtime)
    cache, _ = allocation(runtime, tag="kv_cache", policy=SavePolicy.DISCARD)
    runtime.prepare_cpu_backup("weights")
    runtime.reset_snapshots()
    runtime.sleep(())
    runtime.wake_up(["weights"])
    assert weights.data[0] == 0
    with pytest.raises(RuntimeError):
        runtime.require_awake("inference")
    runtime.wake_up(["kv_cache"])
    assert cache.data[0] == 0
    runtime.require_awake("inference")


def test_disk_demote_and_restore_without_full_host_copy(tmp_path):
    runtime = BackupRuntime(
        FakeBackend(),
        RuntimeConfig(disk_enabled=True, disk_root=tmp_path, chunk_bytes=4096, direct_io=False),
    )
    try:
        device, region = allocation(runtime)
        runtime.prepare_sleep("weights")
        assert runtime.reclaim(region.size_bytes)["released"]
        assert runtime.cpu_backup_pool.reserved_bytes == 0
        runtime.sleep("weights", skip_prepare=True)
        runtime.wake_up()
        assert device.data[0] == 37
        assert runtime.disk_backup_read_bytes_total == region.size_bytes
    finally:
        runtime.close()


def test_corrupt_manifest_fails_before_mapping(tmp_path):
    runtime = BackupRuntime(
        FakeBackend(),
        RuntimeConfig(disk_enabled=True, disk_root=tmp_path, chunk_bytes=4096, direct_io=False),
    )
    try:
        device, region = allocation(runtime)
        runtime.prepare_sleep("weights")
        runtime.reclaim(region.size_bytes)
        runtime.sleep("weights", skip_prepare=True)
        ref = runtime.allocations[region.address].disk_backup_ref
        (ref.bundle_dir / "manifest.json").write_text("{}")
        with pytest.raises(RuntimeError, match="checksum"):
            runtime.wake_up()
        assert not runtime.backend.maps
        assert device.data[0] == 0
    finally:
        runtime.close()


def test_failed_disk_write_keeps_only_restore_source(tmp_path, monkeypatch):
    runtime = BackupRuntime(
        FakeBackend(),
        RuntimeConfig(disk_enabled=True, disk_root=tmp_path, chunk_bytes=4096, direct_io=False),
    )
    try:
        device, region = allocation(runtime)
        runtime.prepare_sleep("weights")

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr(runtime.disk_backup_store, "write_bundle", fail)
        assert not runtime.reclaim(region.size_bytes)["released"]
        assert runtime.cpu_backup_pool.reserved_bytes == region.size_bytes
        runtime.sleep("weights", skip_prepare=True)
        runtime.wake_up()
        assert device.data[0] == 37
    finally:
        runtime.close()


def test_reclaim_waits_for_copy_to_finish(runtime, monkeypatch):
    device, region = allocation(runtime)
    copying, release = threading.Event(), threading.Event()
    original = runtime.backend.copy

    def blocked(*args):
        copying.set()
        assert release.wait(5)
        original(*args)

    monkeypatch.setattr(runtime.backend, "copy", blocked)
    prepare = threading.Thread(target=runtime.prepare_sleep, args=("weights",))
    prepare.start()
    assert copying.wait(5)
    reclaim = threading.Thread(target=runtime.reclaim, args=(region.size_bytes,))
    reclaim.start()
    release.set()
    prepare.join(5)
    reclaim.join(5)
    assert not prepare.is_alive() and not reclaim.is_alive()
    assert runtime.cpu_backup_pool.reserved_bytes == region.size_bytes
    assert device.data[0] == 37


def test_payload_failure_after_remap_requires_recovery(tmp_path):
    value = BackupRuntime(
        FakeBackend(),
        RuntimeConfig(disk_enabled=True, disk_root=tmp_path, chunk_bytes=4096, direct_io=False),
    )
    try:
        device, region = allocation(value)
        value.prepare_sleep("weights")
        value.reclaim(region.size_bytes)
        value.sleep("weights", skip_prepare=True)
        ref = value.allocations[region.address].disk_backup_ref
        (ref.bundle_dir / "data.bin").write_bytes(bytes(region.size_bytes))
        with pytest.raises(RuntimeError, match="checksum"):
            value.wake_up()
        assert value.residency_state == ResidencyState.RECOVERY_REQUIRED
        assert device.data[0] == 0
    finally:
        value.close()


def test_prepare_diagnostics_account_for_mutable_copies(tmp_path):
    import json

    path = tmp_path / "profile.jsonl"
    value = BackupRuntime(FakeBackend(), RuntimeConfig(profile_path=path))
    try:
        device, region = allocation(value, policy=SavePolicy.SNAPSHOT)
        value.prepare_cpu_backup("weights")
        value.prepare_sleep("weights")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        event = next(row for row in events if row["phase"] == "allocator_prepare_sleep")
        assert event["prepared_bytes"] == region.size_bytes
        assert event["reused_bytes"] == 0
        assert device.data[0] == 37
    finally:
        value.close()
