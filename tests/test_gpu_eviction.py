import ctypes

import pytest
from test_runtime import FakeBackend, FakeStream, allocation

from switch_runtime.config import RuntimeConfig
from switch_runtime.contracts import SavePolicy
from switch_runtime.runtime import BackupRuntime, CpuBackupState, ResidencyState


def test_partial_sleep_discards_kv_and_restores_only_missing_regions():
    runtime = BackupRuntime(FakeBackend(), RuntimeConfig())
    first, a = allocation(runtime, size=8192)
    second, b = allocation(runtime, size=4096)
    kv, k = allocation(runtime, size=4096, tag="kv_cache", policy=SavePolicy.DISCARD)
    runtime.sleep("weights", release_bytes=8192)
    assert runtime.partial_sleep
    assert runtime.backend.unmaps == [b.address, k.address]
    assert runtime.allocations[a.address].resident
    with pytest.raises(RuntimeError, match="sleeping"):
        runtime.require_awake("infer")
    # Retained immutable CPU storage may be reclaimed; its GPU bytes remain exact.
    runtime.reclaim(8192)
    assert runtime.allocations[a.address].cpu_backup_buffer is None
    assert runtime.allocations[b.address].cpu_backup_buffer is not None
    before = len(runtime.backend.copies)
    runtime.wake_up()
    assert runtime.backend.maps == [b.address, k.address]
    assert len(runtime.backend.copies) == before + 1
    assert first.data[0] == second.data[0] == 37
    assert kv.data[0] == 0
    assert not runtime.partial_sleep
    runtime.close()


def test_zero_budget_keeps_weights_but_discards_kv_then_full_sleep_works():
    runtime = BackupRuntime(FakeBackend(), RuntimeConfig())
    weight, region = allocation(runtime)
    kv, _ = allocation(runtime, tag="kv_cache", policy=SavePolicy.DISCARD)
    runtime.sleep("weights", release_bytes=0)
    assert weight.data[0] == 37 and kv.data[0] == 0
    runtime.wake_up(["weights"])
    assert runtime.residency_state == ResidencyState.SLEEPING
    runtime.wake_up(["kv_cache"])
    runtime.sleep("weights")
    assert not runtime.allocations[region.address].resident
    runtime.wake_up()
    assert weight.data[0] == 37
    runtime.close()


def test_async_restore_fences_all_copies_before_reclaim_or_readiness():
    backend = FakeBackend()
    runtime = BackupRuntime(backend, RuntimeConfig(async_cpu_restore=True))
    a, _ = allocation(runtime)
    b, _ = allocation(runtime)
    runtime.sleep("weights")

    class DeferredStream(FakeStream):
        pending = []

        def submit(self, slot, source, destination):
            self.pending.append((source, destination))

        def synchronize(self):
            if not self.pending:
                return
            assert runtime.residency_state == ResidencyState.SLEEPING
            assert all(
                d.cpu_backup_state == CpuBackupState.RESTORING_H2D
                for d in runtime.allocations.values()
            )
            assert not runtime.reclaim(8192)["released"]
            for source, destination in self.pending:
                ctypes.memmove(
                    destination, ctypes.addressof(ctypes.c_char.from_buffer(source)), len(source)
                )
            self.pending.clear()

        def close(self):
            self.synchronize()

    backend.restore_stream = lambda slots: DeferredStream()
    runtime.wake_up()
    assert a.data[0] == b.data[0] == 37
    assert runtime.residency_state == ResidencyState.AWAKE
    runtime.close()


def test_async_restore_failure_fences_stream_and_stays_fail_closed():
    backend = FakeBackend()
    runtime = BackupRuntime(backend, RuntimeConfig(async_cpu_restore=True))
    allocation(runtime)
    runtime.sleep("weights")
    fenced = []

    class BrokenStream(FakeStream):
        def submit(self, slot, source, destination):
            raise RuntimeError("uncertain enqueue")

        def close(self):
            fenced.append(True)

    backend.restore_stream = lambda slots: BrokenStream()
    with pytest.raises(RuntimeError, match="uncertain enqueue"):
        runtime.wake_up()
    assert fenced == [True]
    assert runtime.residency_state == ResidencyState.RECOVERY_REQUIRED
    assert not runtime.reclaim(4096)["released"]
    runtime.close()


def test_partial_unmap_failure_rejects_further_inference():
    backend = FakeBackend()
    runtime = BackupRuntime(backend, RuntimeConfig())
    allocation(runtime)
    backend.fail_unmap = True
    with pytest.raises(RuntimeError, match="unmap failed"):
        runtime.sleep("weights", release_bytes=4096)
    assert runtime.residency_state == ResidencyState.RECOVERY_REQUIRED
    runtime.close()
