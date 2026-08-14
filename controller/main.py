import argparse
import asyncio
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI

from controller.backup_pool import BackupPoolState
from controller.config import ControllerConfig, load_config
from controller.memory_pressure import MemoryPressureMonitor
from controller.metrics import MetricsRecorder
from controller.policies import make_policy
from controller.router import make_router
from controller.state import ControllerState
from controller.vllm_client import VLLMClient


def create_app(config: ControllerConfig) -> FastAPI:
    state = ControllerState.from_models(list(config.models), config.controller.startup_awake_model)
    policy = make_policy(config.controller.policy)
    vllm_client = VLLMClient(
        config.models,
        request_timeout_s=config.controller.request_timeout_s,
        switch_timeout_s=config.controller.switch_timeout_s,
    )
    metrics_recorder = MetricsRecorder(config.controller.metrics_path)
    backup_pool = BackupPoolState(
        config.controller.cpu_backup_global_cap_bytes,
        model_priorities=config.controller.cpu_backup_model_priorities,
        default_model_priority=config.controller.cpu_backup_default_model_priority,
    )
    memory_pressure = MemoryPressureMonitor(
        backup_pool,
        reclaim_available_ratio=(config.controller.cpu_memory_reclaim_available_ratio),
        recovery_available_ratio=(config.controller.cpu_memory_recovery_available_ratio),
        reclaim_available_bytes=(config.controller.cpu_memory_reclaim_available_bytes),
        recovery_available_bytes=(config.controller.cpu_memory_recovery_available_bytes),
        poll_interval_s=config.controller.cpu_memory_poll_interval_s,
        consecutive_samples=(config.controller.cpu_memory_pressure_consecutive_samples),
        reclaim_cooldown_s=config.controller.cpu_memory_reclaim_cooldown_s,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        monitor_task: asyncio.Task[None] | None = None
        if memory_pressure.enabled:
            monitor_task = asyncio.create_task(memory_pressure.run())
        try:
            yield
        finally:
            if monitor_task is not None:
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
            await vllm_client.aclose()

    app = FastAPI(
        title="vLLM Model Switch Controller",
        # OpenAPI requires an info.version string; avoid presenting package metadata as a
        # release declaration for the source checkout.
        version="unversioned",
        lifespan=lifespan,
    )
    app.state.controller_config = config
    app.state.controller_state = state
    app.state.vllm_client = vllm_client
    app.state.backup_pool = backup_pool
    app.state.memory_pressure = memory_pressure
    app.include_router(
        make_router(
            config,
            state,
            policy,
            vllm_client,
            metrics_recorder,
            backup_pool,
            memory_pressure,
        )
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vLLM model switch controller")
    parser.add_argument("--config", default="configs/models.launcher.example.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.controller.host, port=config.controller.port)


if __name__ == "__main__":
    main()
