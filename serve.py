import gc
import io
import os
import socket
import sys
import yaml
import json
import argparse
import asyncio
from io import BytesIO
from pathlib import Path
from time import time
from contextlib import contextmanager, nullcontext
from typing import Literal

import torch
import uvicorn
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from loguru import logger
from pydantic import BaseModel, field_validator
from fastapi import FastAPI, UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from loki_logger import LokiLogManager
from prometheus_manager import VictoriaMetricsManager
from settings import settings


class _StderrToLoguru(io.TextIOBase):
    """Intercept stderr writes (e.g. tqdm progress) and route them through Loguru."""

    def __init__(self, task_id: str = "") -> None:
        super().__init__()
        self._task_id = task_id

    def write(self, text: str) -> int:
        text = text.rstrip("\r\n")
        if text:
            logger.debug(format_task_log(self._task_id, text) if self._task_id else text)
        return len(text)

    def flush(self) -> None:
        pass


@contextmanager
def redirect_stderr_to_loguru(task_id: str = ""):
    """Context manager: replace sys.stderr with Loguru for the duration of the block."""
    old_stderr = sys.stderr
    sys.stderr = _StderrToLoguru(task_id=task_id)
    try:
        yield
    finally:
        sys.stderr = old_stderr


REQUIRED_MODELS = {
    "microsoft/TRELLIS.2-4B",
    "microsoft/TRELLIS-image-large",
    "ZhengPeng7/BiRefNet",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def load_model_versions() -> dict[str, str]:
    """Load pinned model versions from model_versions.yml."""
    versions_file = Path(__file__).parent / "model_versions.yml"
    with open(versions_file) as f:
        data = yaml.safe_load(f)["huggingface"]
    
    # Extract revisions and validate
    model_versions = {k: v["revision"] for k, v in data.items()}
    
    if missing := REQUIRED_MODELS - model_versions.keys():
        raise ValueError(f"Missing required models in model_versions.yml: {missing}")
    
    return model_versions


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


class Parameters(BaseModel):
    texture_size: int = 2048
    pipeline_type: str = "1024_cascade"
    face_count: int = 100000

    @field_validator('texture_size')
    @classmethod
    def validate_texture_size(cls, texture_size: int) -> int:
        if texture_size not in (1024, 2048, 4096):
            logger.warning(f"Unsupported texture size. Supported texture sizes: [1024, 2048, 4096]. Default to 2048.")
            texture_size = 2048
        return texture_size

    @field_validator("pipeline_type")
    @classmethod
    def validate_pipeline_type(cls, pipeline_type: str) -> str:
        if pipeline_type not in ("512", "1024", "1024_cascade", "1536_cascade"):
            logger.warning(f"Unsupported 3d pipeline. Supported texture sizes: [512, 1024, 1024_cascade, 1536_cascade]. Default to 1024_cascade.")
            pipeline_type = "1024_cascade"
        return pipeline_type


def format_task_log(task_id: str, message: str) -> str:
    return f"{task_id}: {message}"


def parse_parameters_args(params: dict | None, task_id: str) -> Parameters:
    params = params or {}
    parsed_params = Parameters(**params)

    logger.info(format_task_log(task_id, f"Pipeline Type: {parsed_params.pipeline_type}"))
    logger.info(format_task_log(task_id, f"Texture size: {parsed_params.texture_size}"))
    logger.info(format_task_log(task_id, f"Face count: {parsed_params.face_count}"))

    return parsed_params


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


executor = ThreadPoolExecutor(max_workers=1)


def detect_instance_identity() -> tuple[str, Literal["verda", "runpod"]]:
    if pod_id := os.environ.get("RUNPOD_POD_ID"):
        logger.info(f"Detected RunPod, pod ID: {pod_id}")
        return pod_id, "runpod"
    container_id = socket.gethostname()
    logger.info(f"Detected Verda, container ID: {container_id}")
    return container_id, "verda"


def _build_loki_log_manager(*, generator_mesh_v1_id: str, worker_type: Literal["verda", "runpod"]) -> LokiLogManager | None:
    if not settings.loki_enabled:
        return None
    return LokiLogManager(
        endpoint=settings.loki_endpoint,
        username=settings.loki_username,
        password=settings.loki_password.get_secret_value(),
        generator_mesh_v1_id=generator_mesh_v1_id,
        worker_type=worker_type,
        push_interval_seconds=settings.loki_push_interval_seconds,
        batch_size=settings.loki_batch_size,
        timeout_seconds=settings.loki_timeout_seconds,
    )


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    instance_id, instance_type = detect_instance_identity()
    loki_manager = _build_loki_log_manager(generator_mesh_v1_id=instance_id, worker_type=instance_type)
    loki_sink_id: int | None = None

    async with (
        loki_manager or nullcontext(),
        VictoriaMetricsManager(
            pushgateway_url=settings.prometheus_push_gateway_url,
            username=settings.prometheus_push_gateway_username,
            password=settings.prometheus_push_gateway_password.get_secret_value(),
        ) as victoria_manager,
    ):
        if loki_manager is not None:
            loki_sink_id = logger.add(loki_manager.sink, level="DEBUG", enqueue=True)
            logger.info("Loki log shipping is enabled.")

        app.state.victoria_manager = victoria_manager
        app.state.instance_id = instance_id
        app.state.instance_type = instance_type

        logger.info("Loading Trellis 2 generator models ...")
        try:
            model_versions = load_model_versions()
            logger.info(f"Loaded pinned revisions for {len(model_versions)} models")

            app.state.trellis_generator = Trellis2ImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B",
                model_versions,
            )
            app.state.trellis_generator.to("cuda")

        except Exception as e:
            logger.exception(f"Exception during model loading: {e}")
            raise SystemExit("Model failed to load → exiting server")

        yield

        logger.info("Shutting down...")
        if loki_sink_id is not None:
            logger.remove(loki_sink_id)


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def generation_block(prompt_image: Image.Image, params_dict: dict, seed: int = -1, task_id: str = "") -> BytesIO:
    """ Function for 3D data generation using provided image"""

    with logger.contextualize(task_id=task_id) if task_id else nullcontext(), redirect_stderr_to_loguru(task_id):
        t_start = time()
        parsed_params = parse_parameters_args(params_dict, task_id)

        mesh = app.state.trellis_generator.run(image=prompt_image, seed=seed, pipeline_type=parsed_params.pipeline_type)[0]
        mesh.simplify()

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=parsed_params.face_count,
            texture_size=parsed_params.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True
        )

        buffer = BytesIO()
        glb.export(buffer, extension_webp=False, file_type="glb")
        buffer.seek(0)

        t_get_model = time()
        logger.debug(format_task_log(task_id, f"Model Generation took: {(t_get_model - t_start)} secs."))

        clean_vram()

        t_gc = time()
        logger.debug(format_task_log(task_id, f"Garbage Collection took: {(t_gc - t_get_model)} secs"))

        return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1), params: str|None = Form(None), task_id: str = Form("")) -> Response:
    """ Generates a 3D model as GLB file """

    with logger.contextualize(task_id=task_id) if task_id else nullcontext():
        logger.info(format_task_log(task_id, "Task received. Prompt-Image"))

        contents = await prompt_image_file.read()
        prompt_image = Image.open(BytesIO(contents))

        params_dict = json.loads(params) if params else {}

        loop = asyncio.get_running_loop()
        t_start = time()
        try:
            buffer = await loop.run_in_executor(executor, generation_block, prompt_image, params_dict, seed, task_id)
            generation_time = time() - t_start
            await app.state.victoria_manager.record_generation_metric(
                generation_time=generation_time,
                generator_mesh_v1_id=app.state.instance_id,
                worker_type=app.state.instance_type,
                task_id=task_id,
            )
        except Exception:
            logger.exception(format_task_log(task_id, "Generation failed."))
            await app.state.victoria_manager.record_generation_error_metric(
                generator_mesh_v1_id=app.state.instance_id,
                worker_type=app.state.instance_type,
                task_id=task_id,
            )
            raise

        buffer.seek(0, 2)
        buffer_size = buffer.tell()
        buffer.seek(0)

        logger.info(format_task_log(task_id, "Task completed."))

        async def generate_chunks():
            chunk_size = 1024 * 1024  # 1 MB
            while chunk := buffer.read(chunk_size):
                yield chunk

        clean_vram()

        return StreamingResponse(
            generate_chunks(),
            media_type="application/octet-stream",
            headers={"Content-Length": str(buffer_size)}
        )


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    """ Return if the server is alive """
    return {"status": "healthy"}


if __name__ == "__main__":
    args: argparse.Namespace  = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)