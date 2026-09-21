# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from switch_runtime.disk import (
    DiskRestoreSegment,
    DiskWriteSegment,
    ExactDiskBackupStore,
)


def test_concurrent_store_startup_keeps_both_active_incarnations(tmp_path: Path):
    import concurrent.futures

    def create() -> ExactDiskBackupStore:
        return ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(create)
        second_future = executor.submit(create)
        first = first_future.result()
        second = second_future.result()

    assert first.process_dir.exists()
    assert second.process_dir.exists()
    first.cleanup_process_dir()
    second.cleanup_process_dir()


def test_store_startup_removes_only_unlocked_stale_incarnations(tmp_path: Path):
    stale = tmp_path / "vllm-switch-stale"
    stale.mkdir()
    (stale / "OWNER.lock").write_text("{}\n", encoding="utf-8")

    first = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    assert not stale.exists()
    second = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    assert first.process_dir.exists()
    assert second.process_dir.exists()

    first.cleanup_process_dir()
    second.cleanup_process_dir()


def test_store_startup_removes_legacy_unlocked_stale_incarnation(tmp_path: Path):
    stale = tmp_path / "switch-crashed-worker"
    stale.mkdir()
    (stale / "OWNER.lock").write_text("{}\n", encoding="utf-8")

    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)

    assert not stale.exists()
    store.cleanup_process_dir()


def test_direct_io_rejects_unaligned_buffer_address(tmp_path: Path):
    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=True)
    backing = bytearray(8192)
    unaligned = memoryview(backing)[1:4097]

    with pytest.raises(ValueError, match="buffer address"):
        store.write_bundle(
            [
                DiskWriteSegment(
                    region_id="a",
                    data=unaligned,
                )
            ]
        )


def test_global_restore_pipeline_spans_segments_and_hashes_in_parallel(
    tmp_path: Path,
    monkeypatch,
):
    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    first = bytes(index % 251 for index in range(4096 * 4))
    second = bytes(255 - index % 251 for index in range(4096 * 4))
    refs = store.write_bundle(
        [
            DiskWriteSegment("a", memoryview(first)),
            DiskWriteSegment("b", memoryview(second)),
        ]
    )
    manifest = json.loads((refs["a"].bundle_dir / "manifest.json").read_text())
    assert manifest["magic"] == "vllm-exact-runtime-backup"
    assert store.process_dir.name.startswith("vllm-switch-")
    buffers = tuple(memoryview(bytearray(4096)) for _ in range(4))
    restored = {0x100000: bytearray(len(first)), 0x200000: bytearray(len(second))}
    hash_threads: set[str] = set()
    original_sha256 = hashlib.sha256

    def tracked_sha256(data=b""):
        name = threading.current_thread().name
        if name.startswith("vllm-switch-exact-disk-restore-hash-"):
            hash_threads.add(name)
            threading.Event().wait(0.001)
        return original_sha256(data)

    monkeypatch.setattr(hashlib, "sha256", tracked_sha256)

    def consume(chunk: memoryview, destination: int, _slot: int) -> None:
        base = 0x100000 if destination < 0x200000 else 0x200000
        offset = destination - base
        restored[base][offset : offset + len(chunk)] = chunk

    stats = store.restore_segments_pipelined(
        [
            DiskRestoreSegment(refs["a"], 0x100000),
            DiskRestoreSegment(refs["b"], 0x200000),
        ],
        buffers,
        consume,
        lambda _slot: 0.0,
        hash_workers=2,
    )

    assert bytes(restored[0x100000]) == first
    assert bytes(restored[0x200000]) == second
    assert hash_threads == {
        "vllm-switch-exact-disk-restore-hash-0",
        "vllm-switch-exact-disk-restore-hash-1",
    }
    assert stats["pipeline_depth"] == 4
    assert stats["hash_workers"] == 2
    assert stats["hash_worker_s"] > 0
    assert stats["bytes_read"] == len(first) + len(second)


def test_global_restore_pipeline_rejects_forged_reference(tmp_path: Path):
    import dataclasses

    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    payload = bytes(index % 253 for index in range(8192))
    ref = store.write_bundle([DiskWriteSegment("a", memoryview(payload))])["a"]
    forged = dataclasses.replace(ref, offset_bytes=4096)

    with pytest.raises(RuntimeError, match="reference does not match manifest"):
        store.restore_segments_pipelined(
            [DiskRestoreSegment(forged, 0x100000)],
            [memoryview(bytearray(4096)) for _ in range(4)],
            lambda *_args: None,
            lambda _slot: 0.0,
        )


def test_global_restore_pipeline_fails_closed_on_corruption(tmp_path: Path):
    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    payload = bytes(index % 253 for index in range(8192))
    ref = store.write_bundle([DiskWriteSegment("a", memoryview(payload))])["a"]
    with (ref.bundle_dir / "data.bin").open("r+b") as file:
        file.seek(17)
        file.write(b"\xff")
        file.flush()

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.restore_segments_pipelined(
            [DiskRestoreSegment(ref, 0x100000)],
            [memoryview(bytearray(4096)) for _ in range(4)],
            lambda *_args: None,
            lambda _slot: 0.0,
        )


def test_global_restore_pipeline_cleans_up_partial_thread_start(
    tmp_path: Path,
    monkeypatch,
):
    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    payload = bytes(range(64)) * 256
    ref = store.write_bundle([DiskWriteSegment("a", memoryview(payload))])["a"]
    original_start = threading.Thread.start
    starts = 0

    def fail_second_start(thread):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("thread start failed")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_second_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        store.restore_segments_pipelined(
            [DiskRestoreSegment(ref, 0x100000)],
            [memoryview(bytearray(4096)) for _ in range(4)],
            lambda _chunk, _destination, _slot: None,
            lambda _slot: 0.0,
            hash_workers=2,
        )

    assert not any(
        thread.name.startswith("exact-disk-restore-") for thread in threading.enumerate()
    )


def test_republishing_generation_removes_superseded_bundle(tmp_path: Path):
    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    first = store.write_bundle(
        [
            DiskWriteSegment(
                region_id="a",
                data=memoryview(bytes(4096)),
            )
        ]
    )["a"]
    second = store.write_bundle(
        [
            DiskWriteSegment(
                region_id="a",
                data=memoryview(bytes([1]) * 4096),
            )
        ]
    )["a"]

    store.delete_bundle(first)

    assert not first.bundle_dir.exists()
    assert second.bundle_dir.exists()
    assert len(list(store.process_dir.glob("*.ready"))) == 1


def test_failed_publication_preserves_previous_bundle(tmp_path: Path, monkeypatch):
    store = ExactDiskBackupStore(tmp_path, chunk_bytes=4096, direct_io=False)
    original = store.write_bundle(
        [
            DiskWriteSegment(
                region_id="a",
                data=memoryview(bytes(4096)),
            )
        ]
    )["a"]

    def fail_publish(*_args):
        raise OSError("publish failed")

    monkeypatch.setattr(store, "_publish_bundle", fail_publish)

    with pytest.raises(OSError, match="publish failed"):
        store.write_bundle(
            [
                DiskWriteSegment(
                    region_id="a",
                    data=memoryview(bytes([1]) * 4096),
                )
            ]
        )

    assert original.bundle_dir.is_dir()
    assert len(list(store.process_dir.glob("*.ready"))) == 1
    assert not list(store.process_dir.glob("*.tmp"))
