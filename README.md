# 404-base-miner (Trellis 2, commercially ready to use)

<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

Current base miner implementation is based on recently released TRELLIS 2 mesh generation model.
**TRELLIS.2** is a state-of-the-art large 3D generative model (4B parameters) designed for high-fidelity 
**image-to-3D** generation. It leverages a novel "field-free" sparse voxel structure termed 
**O-Voxel** to reconstruct and generate arbitrary 3D assets with complex topologies, sharp features, 
and full PBR materials.

### 🛠️ Hardware Requirements

To run this generator you will need a GPU with at least 48 GB of VRAM. It can work with GPUs from NVIDIA Blackwell family.
You can run it on Geforce 5090 RTX if the generation settings are set to 512 resolution when you call **run(...)** method 
(see **serve.py**).

### 🛠️ Software Requirements
- latest docker package (we provide docker file in "docker" folder) or latest conda environment (we provide "conda_env.yml");
- NVIDIA GPU with cuda 12.8 support
- python 3.11

### 🔑 Huggingface Token Requirement
The code needs access to the gated model (commercially compliant) on huggingface: [https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
### Installation

- Docker (building & pushing to remote register):
```console
cd /docker
docker build --build-arg GITHUB_USER="" --build-arg GITHUB_TOKEN="" -t docker_name:docker-tag .
docker tag docker_name:docker-tag docker-register-path:docker-register-name
docker push docker-register-path:docker-register-name   
```
- Conda Env. (shell script will install everything you need to run the project):
```console
# to install conda env
bash setup_env.sh

# to uninstall conda env
bash cleanup_env.sh
```

### 🚀 Usage

### How to run:
- Docker (run locally):

**Build the image:**
```bash
# Option 1: Using secret files (recommended)
echo "your_github_username" > github_user.txt
echo "your_github_token" > github_token.txt

DOCKER_BUILDKIT=1 docker build \
  --secret id=github_user,src=github_user.txt \
  --secret id=github_token,src=github_token.txt \
  -t trellis2-gen \
  -f docker/Dockerfile .

rm github_user.txt github_token.txt

# Option 2: Using environment variables
export GITHUB_USER="your_github_username"
export GITHUB_TOKEN="your_github_token"

DOCKER_BUILDKIT=1 docker build \
  --secret id=github_user,env=GITHUB_USER \
  --secret id=github_token,env=GITHUB_TOKEN \
  -t trellis2-gen \
  -f docker/Dockerfile .
```

**Run the container:**
```bash
docker run --gpus all -p 10006:10006 trellis2-gen
```

**Test the API:**
```bash
curl -X POST "http://0.0.0.0:10006/generate" -F prompt_image_file=@sample_image.png -o sample_model.glb
```

- Conda Env.:
```commandline
# start pm2 process
pm2 start generation.config.js

# view logs
pm2 logs

# send prompt image
curl -X POST "http://0.0.0.0:10006/generate" -F prompt_image_file=@sample_image.png -o sample_model.glb
```

## ⚖️ License

This model and code are released under the **[MIT License](LICENSE)**.
