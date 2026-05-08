import asyncio
import json
import threading
from collections import defaultdict
from typing import Literal

import httpx

from constants import SERVICE_NAME


class LokiLogManager:
    """Collect Loguru records and periodically push them to Loki."""

    def __init__(
        self,
        *,
        endpoint: str,
        username: str,
        password: str,
        worker_id: str,
        worker_type: Literal["verda", "runpod"],
        push_interval_seconds: float,
        batch_size: int,
        timeout_seconds: float,
    ) -> None:
        self._endpoint = endpoint
        self._auth = (username, password)
        self._worker_id = worker_id
        self._worker_type = worker_type
        self._push_interval_seconds = max(push_interval_seconds, 0.1)
        self._batch_size = max(batch_size, 1)
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._lock = threading.Lock()
        self._buffer: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
        self._flush_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def __aenter__(self) -> "LokiLogManager":
        self._flush_task = asyncio.create_task(self._run_flush_loop())
        return self

    async def __aexit__(self, *args: object) -> None:
        self._stop_event.set()
        if self._flush_task is not None:
            await self._flush_task
        await self.flush()
        await self._client.aclose()

    def sink(self, message: object) -> None:
        """Loguru sink callback used by `logger.add()`."""
        try:
            record = message.record  # type: ignore[attr-defined]
            level = str(record["level"].name).lower()
            logger_name = str(record["name"])
            ts_ns = str(int(record["time"].timestamp() * 1_000_000_000))
            log_line = str(record["message"])
            key = (
                f"service={SERVICE_NAME}",
                f"type={self._worker_type}",
                f"worker_id={self._worker_id}",
                f"level={level}",
                f"logger={logger_name}",
            )
        except Exception:
            # Never interrupt app execution because of logging export path.
            return

        with self._lock:
            self._buffer[key].append((ts_ns, log_line))

    async def _run_flush_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._push_interval_seconds)
            except TimeoutError:
                pass
            await self.flush()

    async def flush(self) -> None:
        streams = self._drain_batch()
        if not streams:
            return

        payload = {"streams": streams}
        try:
            response = await self._client.post(
                self._endpoint,
                auth=self._auth,
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except Exception:
            # Avoid using Loguru here to prevent recursive sink calls.
            return

    def _drain_batch(self) -> list[dict[str, object]]:
        with self._lock:
            streams: list[dict[str, object]] = []
            for key, values in list(self._buffer.items()):
                if not values:
                    continue
                taken = values[: self._batch_size]
                self._buffer[key] = values[self._batch_size :]
                if not self._buffer[key]:
                    del self._buffer[key]
                labels = {}
                for item in key:
                    label_key, label_value = item.split("=", maxsplit=1)
                    labels[label_key] = label_value
                streams.append({"stream": labels, "values": taken})
            return streams
