import gc
import argparse
import asyncio
import os
from io import BytesIO
from pathlib import Path
from time import time
from typing import Optional

import yaml
import torch
import uvicorn
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline


def load_model_versions() -> dict:
    """Load pinned model versions from model_versions.yml."""
    versions_file = Path(__file__).parent / "model_versions.yml"
    if versions_file.exists():
        with open(versions_file) as f:
            return yaml.safe_load(f) or {}
    logger.warning("model_versions.yml not found, using unpinned versions")
    return {}


def get_hf_revision(model_id: str) -> Optional[str]:
    """Get the pinned HuggingFace revision for a model, or None if not pinned."""
    versions = load_model_versions()
    hf_models = versions.get("huggingface", {})
    model_config = hf_models.get(model_id, {})
    revision = model_config.get("revision") if isinstance(model_config, dict) else None
    if revision:
        logger.info(f"Using pinned revision for {model_id}: {revision}")
    else:
        logger.warning(f"No pinned revision for {model_id}, using latest")
    return revision


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


executor = ThreadPoolExecutor(max_workers=1)

class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    logger.info("Loading Trellis 2 generator models ...")
    try:
        # Get pinned revisions from model_versions.yml
        trellis_revision = get_hf_revision("microsoft/TRELLIS.2-4B")
        birefnet_revision = get_hf_revision("ZhengPeng7/BiRefNet")
        dinov3_revision = get_hf_revision("facebook/dinov3-vitl16-pretrain-lvd1689m")
        
        # Build model_revisions dict for external models referenced in pipeline.json
        model_revisions = {}
        trellis_image_large_revision = get_hf_revision("microsoft/TRELLIS-image-large")
        if trellis_image_large_revision:
            model_revisions["microsoft/TRELLIS-image-large"] = trellis_image_large_revision
        
        app.state.trellis_generator = Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
            revision=trellis_revision,
            birefnet_revision=birefnet_revision,
            dinov3_revision=dinov3_revision,
            model_revisions=model_revisions,
        )
        app.state.trellis_generator.to("cuda")

    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        raise SystemExit("Model failed to load → exiting server")

    yield


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def generation_block(prompt_image: Image.Image, seed: int = -1):
    """ Function for 3D data generation using provided image"""

    t_start = time()
    mesh = app.state.trellis_generator.run(image=prompt_image, seed=seed, pipeline_type="1024_cascade")[0]
    mesh.simplify()

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=1024,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True
    )

    buffer = BytesIO()
    glb.export(buffer, extension_webp=False, file_type="glb")
    buffer.seek(0)

    t_get_model = time()
    logger.debug(f"Model Generation took: {(t_get_model - t_start)} secs.")

    clean_vram()

    t_gc = time()
    logger.debug(f"Garbage Collection took: {(t_gc - t_get_model)} secs")

    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    """ Generates a 3D model as GLB file """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
    buffer_size = len(buffer.getvalue())
    buffer.seek(0)
    logger.info(f"Task completed.")

    async def generate_chunks():
        chunk_size = 1024 * 1024  # 1 MB
        while chunk := buffer.read(chunk_size):
            yield chunk

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