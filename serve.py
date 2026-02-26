import gc
import yaml
import json
import argparse
import asyncio
from io import BytesIO
from pathlib import Path
from time import time


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


def parse_parameters_args(params: dict | None) -> tuple[int, int, str]:
    params = params or {}

    face_count = params.get("face_count", 100000)

    texture_size = params.get("texture_size", 2048)
    if texture_size not in (1024, 2048, 4096):
        texture_size = 2048

    pipeline_type = params.get("pipeline_type", "1536_cascade")
    if pipeline_type not in ("512", "1024", "1024_cascade", "1536_cascade"):
        pipeline_type = "1536_cascade"

    logger.info(f"Pipeline Type: {pipeline_type}")
    logger.info(f"Texture size: {texture_size}")
    logger.info(f"Face count: {face_count}")

    return face_count, texture_size, pipeline_type


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


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def generation_block(prompt_image: Image.Image, params_dict:dict, seed: int = -1) -> BytesIO:
    """ Function for 3D data generation using provided image"""

    t_start = time()
    face_count, texture_size, pipeline_type = parse_parameters_args(params_dict)

    mesh = app.state.trellis_generator.run(image=prompt_image, seed=seed, pipeline_type=pipeline_type)[0]
    mesh.simplify()

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=face_count,
        texture_size=texture_size,
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
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1), params: str|None = Form(None)) -> Response:
    """ Generates a 3D model as GLB file """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    params_dict = json.loads(params) if params else {}

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, params_dict, seed)
    # buffer_size = len(buffer.getvalue())
    # buffer.seek(0)

    buffer.seek(0, 2)
    buffer_size = buffer.tell()
    buffer.seek(0)

    logger.info(f"Task completed.")

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