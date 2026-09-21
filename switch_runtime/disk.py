# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local exact runtime-weight snapshots backed by a filesystem.

The store deliberately knows nothing about model tensors or CUDA. It publishes a
bundle of allocator-segment bytes transactionally and streams a selected segment
through caller-provided bounded staging storage during restore.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import fcntl
import hashlib
import json
import os
import queue
import shutil
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 2
_ALIGNMENT = 4096


@dataclasses.dataclass(frozen=True)
class DiskWriteSegment:
    region_id: str
    data: memoryview


@dataclasses.dataclass(frozen=True)
class DiskSegmentRef:
    bundle_dir: Path
    region_id: str
    offset_bytes: int
    size_bytes: int
    chunk_bytes: int
    chunk_sha256: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DiskRestoreSegment:
    ref: DiskSegmentRef
    destination_ptr: int


@dataclasses.dataclass(frozen=True)
class _RestoreChunk:
    sequence: int
    slot: int
    destination_ptr: int
    destination_offset: int
    size: int
    expected_sha256: str


class ExactDiskBackupStore:
    """Own immutable exact-byte bundles for one worker-process incarnation."""

    def __init__(
        self,
        root: str | Path,
        *,
        chunk_bytes: int,
        direct_io: bool = True,
    ) -> None:
        if chunk_bytes <= 0 or chunk_bytes % _ALIGNMENT:
            raise ValueError("chunk_bytes must be a positive 4 KiB multiple")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.chunk_bytes = chunk_bytes
        self.direct_io = direct_io
        self._root_lock = (self.root / ".exact-disk-gc.lock").open("a+")
        fcntl.flock(self._root_lock.fileno(), fcntl.LOCK_EX)
        try:
            self._cleanup_stale_process_dirs()
            incarnation = f"vllm-switch-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
            self.process_dir = self.root / incarnation
            self.process_dir.mkdir(mode=0o700, exist_ok=False)
            self._owner_lock = (self.process_dir / "OWNER.lock").open("w+")
            fcntl.flock(self._owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._owner_lock.write(
                json.dumps(
                    {
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "created_ns": time.time_ns(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            self._owner_lock.flush()
        finally:
            fcntl.flock(self._root_lock.fileno(), fcntl.LOCK_UN)

    def _cleanup_stale_process_dirs(self) -> None:
        # Recognize the pre-rename prefix while cleaning stale stores so a
        # crashed older worker cannot strand a multi-gigabyte disk bundle.
        for prefix in ("vllm-switch-*", "switch-*"):
            for process_dir in self.root.glob(prefix):
                owner = process_dir / "OWNER.lock"
                try:
                    lock = owner.open("a+")
                except OSError:
                    continue
                try:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    shutil.rmtree(process_dir, ignore_errors=True)
                finally:
                    lock.close()

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _write_all(fd: int, data: memoryview) -> None:
        written = 0
        while written < len(data):
            try:
                count = os.write(fd, data[written:])
            except InterruptedError:
                continue
            if count <= 0:
                raise OSError("short write while publishing exact disk backup")
            written += count

    @staticmethod
    def _buffer_address(buffer: memoryview) -> int:
        if buffer.readonly:
            raise ValueError("direct-I/O buffer must be writable")
        return ctypes.addressof(ctypes.c_char.from_buffer(buffer))

    def _validate_direct_io_buffer(self, buffer: memoryview) -> None:
        if len(buffer) % _ALIGNMENT:
            raise ValueError("direct-I/O buffer size must be 4 KiB aligned")
        if self._buffer_address(buffer) % _ALIGNMENT:
            raise ValueError("direct-I/O buffer address must be 4 KiB aligned")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _publish_bundle(self, temporary: Path, ready: Path) -> None:
        os.rename(temporary, ready)
        self._fsync_directory(self.process_dir)

    def write_bundle(self, segments: Sequence[DiskWriteSegment]) -> dict[str, DiskSegmentRef]:
        if not segments:
            raise ValueError("an exact disk backup bundle must contain a segment")
        pointers = [segment.region_id for segment in segments]
        if len(pointers) != len(set(pointers)):
            raise ValueError("region IDs must be unique within a bundle")
        if self.direct_io:
            for segment in segments:
                self._validate_direct_io_buffer(segment.data)

        bundle_id = uuid.uuid4().hex
        temporary = self.process_dir / f"{bundle_id}.tmp"
        ready = self.process_dir / f"{bundle_id}.ready"
        temporary.mkdir(mode=0o700)
        manifest_segments: list[dict[str, Any]] = []
        payload_size = sum(len(segment.data) for segment in segments)
        data_path = temporary / "data.bin"
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if self.direct_io:
                flags |= os.O_DIRECT
            fd = os.open(data_path, flags, 0o600)
            try:
                os.posix_fallocate(fd, 0, payload_size)
                offset = 0
                for segment in segments:
                    if os.lseek(fd, offset, os.SEEK_SET) != offset:
                        raise OSError("failed to seek exact disk backup payload")
                    chunk_hashes: list[str] = []
                    for chunk_offset in range(0, len(segment.data), self.chunk_bytes):
                        chunk = segment.data[chunk_offset : chunk_offset + self.chunk_bytes]
                        self._write_all(fd, chunk)
                        chunk_hashes.append(hashlib.sha256(chunk).hexdigest())
                    manifest_segments.append(
                        {
                            "region_id": segment.region_id,
                            "offset_bytes": offset,
                            "size_bytes": len(segment.data),
                            "chunk_sha256": chunk_hashes,
                        }
                    )
                    offset += len(segment.data)
                os.fsync(fd)
            finally:
                os.close(fd)

            manifest: dict[str, Any] = {
                "magic": "vllm-exact-runtime-backup",
                "schema_version": _SCHEMA_VERSION,
                "chunk_bytes": self.chunk_bytes,
                "payload_size_bytes": payload_size,
                "segments": manifest_segments,
            }
            manifest_bytes = self._canonical_json(manifest)
            manifest_path = temporary / "manifest.json"
            manifest_path.write_bytes(manifest_bytes)
            os.chmod(manifest_path, 0o600)
            with manifest_path.open("rb") as file:
                os.fsync(file.fileno())
            commit_path = temporary / "COMMIT"
            commit_path.write_text(
                hashlib.sha256(manifest_bytes).hexdigest() + "\n",
                encoding="ascii",
            )
            os.chmod(commit_path, 0o600)
            with commit_path.open("rb") as file:
                os.fsync(file.fileno())
            self._fsync_directory(temporary)
            self._publish_bundle(temporary, ready)
        except BaseException:
            if ready.exists():
                shutil.rmtree(ready, ignore_errors=True)
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return {
            str(entry["region_id"]): DiskSegmentRef(
                bundle_dir=ready,
                region_id=str(entry["region_id"]),
                offset_bytes=int(entry["offset_bytes"]),
                size_bytes=int(entry["size_bytes"]),
                chunk_bytes=self.chunk_bytes,
                chunk_sha256=tuple(entry["chunk_sha256"]),
            )
            for entry in manifest_segments
        }

    @staticmethod
    def _read_exact_into(fd: int, destination: memoryview, offset: int) -> None:
        read_bytes = 0
        while read_bytes < len(destination):
            try:
                count = os.preadv(
                    fd,
                    [destination[read_bytes:]],
                    offset + read_bytes,
                )
            except InterruptedError:
                continue
            if count <= 0:
                raise RuntimeError("short read from exact disk backup")
            read_bytes += count

    def validate_restore_ref(self, ref: DiskSegmentRef) -> None:
        """Validate committed metadata without reading the payload."""
        manifest_bytes = (ref.bundle_dir / "manifest.json").read_bytes()
        commit = (ref.bundle_dir / "COMMIT").read_text(encoding="ascii").strip()
        if hashlib.sha256(manifest_bytes).hexdigest() != commit:
            raise RuntimeError("manifest checksum mismatch for exact disk backup")
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("unsupported exact disk backup schema")
        if manifest.get("chunk_bytes") != ref.chunk_bytes:
            raise RuntimeError("exact disk backup chunk size does not match manifest")
        matching = [
            segment
            for segment in manifest.get("segments", [])
            if segment.get("region_id") == ref.region_id
        ]
        if len(matching) != 1:
            raise RuntimeError("exact disk backup segment is missing or ambiguous")
        segment = matching[0]
        expected = {
            "offset_bytes": ref.offset_bytes,
            "size_bytes": ref.size_bytes,
            "chunk_sha256": list(ref.chunk_sha256),
        }
        if any(segment.get(field) != value for field, value in expected.items()):
            raise RuntimeError("exact disk backup reference does not match manifest")

    def restore_segments_pipelined(
        self,
        segments: Sequence[DiskRestoreSegment],
        staging: Sequence[memoryview],
        consume: Callable[[memoryview, int, int], None],
        wait_for_consume: Callable[[int], float],
        *,
        hash_workers: int = 2,
    ) -> dict[str, int | float | bool]:
        """Restore a global segment plan through read, hash, and H2D stages."""
        if len(staging) < 3:
            raise ValueError("global exact disk restore requires at least three buffers")
        if hash_workers < 1:
            raise ValueError("global exact disk restore requires a hash worker")
        if not segments:
            return {
                "bytes_read": 0,
                "read_s": 0.0,
                "hash_s": 0.0,
                "consume_s": 0.0,
                "consume_wait_s": 0.0,
                "consume_device_s": 0.0,
                "wall_s": 0.0,
                "checksum_verified": True,
                "direct_io": self.direct_io,
                "pipeline_depth": len(staging),
                "hash_workers": hash_workers,
            }
        for segment in segments:
            self.validate_restore_ref(segment.ref)
            for buffer in staging:
                if len(buffer) < min(segment.ref.chunk_bytes, segment.ref.size_bytes):
                    raise ValueError("staging buffer is smaller than the configured chunk")
                if self.direct_io:
                    self._validate_direct_io_buffer(buffer)

        free_slots: queue.Queue[int] = queue.Queue()
        hash_queue: queue.Queue[_RestoreChunk | None] = queue.Queue()
        ready_queue: queue.Queue[_RestoreChunk | None] = queue.Queue()
        for slot in range(len(staging)):
            free_slots.put(slot)
        cancelled = threading.Event()
        errors: queue.Queue[BaseException] = queue.Queue()
        read_s = 0.0
        hash_s = 0.0
        timing_lock = threading.Lock()
        file_descriptors: dict[Path, int] = {}
        flags = os.O_RDONLY | (os.O_DIRECT if self.direct_io else 0)

        def fail(exc: BaseException) -> None:
            if errors.empty():
                errors.put(exc)
            cancelled.set()

        def reader() -> None:
            nonlocal read_s
            sequence = 0
            try:
                for segment in segments:
                    ref = segment.ref
                    fd = file_descriptors.get(ref.bundle_dir)
                    if fd is None:
                        fd = os.open(ref.bundle_dir / "data.bin", flags)
                        file_descriptors[ref.bundle_dir] = fd
                    for index, expected_hash in enumerate(ref.chunk_sha256):
                        if cancelled.is_set():
                            return
                        while True:
                            if cancelled.is_set():
                                return
                            try:
                                slot = free_slots.get(timeout=0.1)
                                break
                            except queue.Empty:
                                continue
                        destination_offset = index * ref.chunk_bytes
                        size = min(
                            ref.chunk_bytes,
                            ref.size_bytes - destination_offset,
                        )
                        chunk = staging[slot][:size]
                        started = time.perf_counter()
                        self._read_exact_into(
                            fd,
                            chunk,
                            ref.offset_bytes + destination_offset,
                        )
                        elapsed = time.perf_counter() - started
                        with timing_lock:
                            read_s += elapsed
                        hash_queue.put(
                            _RestoreChunk(
                                sequence=sequence,
                                slot=slot,
                                destination_ptr=segment.destination_ptr,
                                destination_offset=destination_offset,
                                size=size,
                                expected_sha256=expected_hash,
                            )
                        )
                        sequence += 1
            except BaseException as exc:
                fail(exc)
            finally:
                for _ in range(hash_workers):
                    hash_queue.put(None)

        def hasher() -> None:
            nonlocal hash_s
            try:
                while True:
                    if cancelled.is_set():
                        return
                    try:
                        entry = hash_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if entry is None:
                        return
                    started = time.perf_counter()
                    actual = hashlib.sha256(staging[entry.slot][: entry.size]).hexdigest()
                    elapsed = time.perf_counter() - started
                    with timing_lock:
                        hash_s += elapsed
                    if actual != entry.expected_sha256:
                        raise RuntimeError("payload checksum mismatch in exact disk backup")
                    ready_queue.put(entry)
            except BaseException as exc:
                fail(exc)
            finally:
                ready_queue.put(None)

        started_at = time.perf_counter()
        reader_thread = threading.Thread(
            target=reader,
            name="vllm-switch-exact-disk-restore-reader",
            daemon=True,
        )
        hash_threads = [
            threading.Thread(
                target=hasher,
                name=f"vllm-switch-exact-disk-restore-hash-{index}",
                daemon=True,
            )
            for index in range(hash_workers)
        ]
        started_threads: list[threading.Thread] = []
        pending_slots: set[int] = set()
        completed_hash_workers = 0
        reorder: dict[int, _RestoreChunk] = {}
        next_sequence = 0
        bytes_read = 0
        consume_s = 0.0
        wait_s = 0.0
        consume_device_s = 0.0
        try:
            # Start every worker inside the cleanup domain. Hash consumers start
            # first so a live reader never fills all slots while a later
            # thread start fails.
            for thread in hash_threads:
                thread.start()
                started_threads.append(thread)
            reader_thread.start()
            started_threads.append(reader_thread)
            while completed_hash_workers < hash_workers:
                if cancelled.is_set() and not errors.empty():
                    raise errors.get()
                try:
                    entry = ready_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if entry is None:
                    completed_hash_workers += 1
                    continue
                reorder[entry.sequence] = entry
                while next_sequence in reorder:
                    current = reorder.pop(next_sequence)
                    slot = current.slot
                    pending_slots.add(slot)
                    consume_started = time.perf_counter()
                    consume(
                        staging[slot][: current.size],
                        current.destination_ptr + current.destination_offset,
                        slot,
                    )
                    consume_s += time.perf_counter() - consume_started
                    bytes_read += current.size
                    wait_started = time.perf_counter()
                    consume_device_s += wait_for_consume(slot)
                    wait_s += time.perf_counter() - wait_started
                    pending_slots.remove(slot)
                    free_slots.put(slot)
                    next_sequence += 1
            if not errors.empty():
                raise errors.get()
            if reorder:
                raise RuntimeError("exact disk restore hash stage lost chunk ordering")
        except BaseException as restore_error:
            cancelled.set()
            for slot in tuple(pending_slots):
                with contextlib.suppress(BaseException):
                    wait_for_consume(slot)
            for slot in range(len(staging)):
                free_slots.put(slot)
            for thread in started_threads:
                thread.join(timeout=5.0)
            if any(thread.is_alive() for thread in started_threads):
                raise RuntimeError("exact disk restore workers failed to stop") from restore_error
            raise
        finally:
            cancelled.set()
            for slot in range(len(staging)):
                free_slots.put(slot)
            for thread in started_threads:
                thread.join(timeout=5.0)
            for fd in file_descriptors.values():
                os.close(fd)
        return {
            "bytes_read": bytes_read,
            "read_s": read_s,
            "hash_worker_s": hash_s,
            "consume_s": consume_s,
            "consume_wait_s": wait_s,
            "consume_device_s": consume_device_s,
            "wall_s": time.perf_counter() - started_at,
            "checksum_verified": True,
            "direct_io": self.direct_io,
            "pipeline_depth": len(staging),
            "hash_workers": hash_workers,
        }

    def cleanup_process_dir(self) -> None:
        if not self._owner_lock.closed:
            fcntl.flock(self._owner_lock.fileno(), fcntl.LOCK_UN)
            self._owner_lock.close()
        shutil.rmtree(self.process_dir, ignore_errors=True)
        if not self._root_lock.closed:
            self._root_lock.close()

    def delete_bundle(self, bundle: DiskSegmentRef | Path) -> None:
        bundle_dir = bundle.bundle_dir if isinstance(bundle, DiskSegmentRef) else bundle
        shutil.rmtree(bundle_dir, ignore_errors=True)
