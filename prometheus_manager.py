import time
from typing import Literal

import httpx
from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram

from constants import SERVICE_NAME


class VictoriaMetricsManager:
    def __init__(
        self,
        *,
        pushgateway_url: str,
        username: str,
        password: str,
    ) -> None:
        self._pushgateway_url = pushgateway_url
        self._username = username
        self._password = password
        self._client = httpx.AsyncClient()
        self._registry = CollectorRegistry()
        self._histograms: dict[str, Histogram] = {}
        self._counters: dict[str, Counter] = {}

        label_names = ["service", "type", "generator_mesh_v1_id", "task_id"]
        error_label_names = ["service", "type", "generator_mesh_v1_id", "task_id", "prompt_url"]

        self._counters["generation_count"] = Counter(
            "generation_count",
            "Number of generation requests",
            labelnames=label_names,
            registry=self._registry,
        )
        self._histograms["generation_latency"] = Histogram(
            "generation_latency",
            "Latency of generation requests in seconds",
            labelnames=label_names,
            registry=self._registry,
        )
        self._counters["generation_error_count"] = Counter(
            "generation_error_count",
            "Number of generation errors",
            labelnames=error_label_names,
            registry=self._registry,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "VictoriaMetricsManager":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    @staticmethod
    def _escape_label_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    async def _push_registry(self) -> None:
        if not self._pushgateway_url:
            return

        metrics_lines: list[str] = []
        for metric in self._registry.collect():
            for sample in metric.samples:
                labels_str = ",".join([f'{k}="{self._escape_label_value(str(v))}"' for k, v in sample.labels.items()])
                metrics_lines.append(f"{sample.name}{{{labels_str}}} {sample.value} {int(time.time() * 1000)}")

        metrics_data = ("\n".join(metrics_lines) + "\n").encode() if metrics_lines else b""

        response = await self._client.post(
            self._pushgateway_url,
            content=metrics_data,
            auth=(self._username, self._password),
            headers={"Content-Type": "text/plain"},
        )
        response.raise_for_status()

    async def record_generation_metric(
        self,
        *,
        generation_time: float,
        generator_mesh_v1_id: str,
        worker_type: Literal["verda", "runpod"],
        task_id: str,
    ) -> None:
        try:
            labels = {
                "service": SERVICE_NAME,
                "type": worker_type,
                "generator_mesh_v1_id": generator_mesh_v1_id,
                "task_id": task_id,
            }
            self._counters["generation_count"].labels(**labels).inc()
            self._histograms["generation_latency"].labels(**labels).observe(generation_time)
            await self._push_registry()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to push metrics to VictoriaMetrics: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error pushing metrics to VictoriaMetrics: {e}")

    async def record_generation_error_metric(
        self,
        *,
        generator_mesh_v1_id: str,
        worker_type: Literal["verda", "runpod"],
        task_id: str,
        prompt_url: str = "",
    ) -> None:
        try:
            labels = {
                "service": SERVICE_NAME,
                "type": worker_type,
                "generator_mesh_v1_id": generator_mesh_v1_id,
                "task_id": task_id,
                "prompt_url": prompt_url,
            }
            self._counters["generation_error_count"].labels(**labels).inc()
            await self._push_registry()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to push metrics to VictoriaMetrics: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error pushing metrics to VictoriaMetrics: {e}")
