from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CPU_BACKUP_PROTOCOL_VERSION = 1
CPU_BACKUP_CAPABILITIES = frozenset(
    {
        "cumulative-release-v1",
        "exact-disk-accounting-v1",
        "process-incarnation-v1",
        "released-bytes-total-v1",
    }
)
CPU_BACKUP_REQUIRED_CAPABILITIES = frozenset(
    {
        "cumulative-release-v1",
        "process-incarnation-v1",
        "released-bytes-total-v1",
    }
)


class CpuBackupProtocolRequest(BaseModel):
    """Versioned metadata shared by worker-to-controller requests."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1]
    capabilities: list[str]

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        if any(not capability.strip() for capability in value):
            raise ValueError("capabilities entries must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("capabilities entries must be unique")
        unknown = set(value) - CPU_BACKUP_CAPABILITIES
        if unknown:
            raise ValueError(f"unsupported CPU backup capabilities: {sorted(unknown)}")
        missing = CPU_BACKUP_REQUIRED_CAPABILITIES - set(value)
        if missing:
            raise ValueError(f"missing required CPU backup capabilities: {sorted(missing)}")
        return value


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "vllm-switch-controller"


class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class BackupRegisterRequest(CpuBackupProtocolRequest):
    client_id: str
    pid: int | None = None
    engine: str = "unknown"
    model_id: str | None = None
    gpu_uuid: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackupUsageRequest(CpuBackupProtocolRequest):
    """Worker-owned aggregate RAM and exact-disk backup accounting.

    Disk bytes are separate telemetry rather than extra RAM state buckets.
    ``ram_reclaimable_with_disk_bytes`` is the reported subset of required RAM
    for which the worker currently has an exact disk restore source.
    """

    client_id: str
    pid: int | None = None
    engine: str = "unknown"
    model_id: str | None = None
    gpu_uuid: str | None = None
    total_bytes: int = Field(ge=0)
    released_bytes_total: int = Field(ge=0)
    required_for_restore_bytes: int = Field(ge=0)
    cache_only_bytes: int = Field(default=0, ge=0)
    invalid_bytes: int = Field(default=0, ge=0)
    free_local_bytes: int = Field(default=0, ge=0)
    disk_backup_current_bytes: int = Field(default=0, ge=0)
    disk_backup_reserved_bytes: int = Field(default=0, ge=0)
    ram_reclaimable_with_disk_bytes: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_accounting(self) -> "BackupUsageRequest":
        accounted = (
            self.required_for_restore_bytes
            + self.cache_only_bytes
            + self.invalid_bytes
            + self.free_local_bytes
        )
        if accounted != self.total_bytes:
            raise ValueError("accounted backup bytes must equal total_bytes")
        if self.ram_reclaimable_with_disk_bytes > self.required_for_restore_bytes:
            raise ValueError(
                "ram_reclaimable_with_disk_bytes cannot exceed required_for_restore_bytes"
            )
        if (
            self.ram_reclaimable_with_disk_bytes > 0
            and self.disk_backup_current_bytes == 0
            and self.disk_backup_reserved_bytes == 0
        ):
            raise ValueError("ram_reclaimable_with_disk_bytes requires a reported disk source")
        disk_fields_present = (
            self.disk_backup_current_bytes > 0
            or self.disk_backup_reserved_bytes > 0
            or self.ram_reclaimable_with_disk_bytes > 0
        )
        if disk_fields_present and "exact-disk-accounting-v1" not in self.capabilities:
            raise ValueError("exact disk accounting requires capability exact-disk-accounting-v1")
        if (
            self.released_bytes_total is not None
            and "released-bytes-total-v1" not in self.capabilities
        ):
            raise ValueError("released_bytes_total requires capability released-bytes-total-v1")
        return self


class BackupReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    target_free_bytes: int = Field(ge=0)
