# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Step1X-3D is a two-stage 3D asset generation framework that creates high-fidelity textured 3D objects from single images. The system combines geometry generation (using VAE-DiT architecture) with texture synthesis (using SD-XL) to produce watertight meshes with consistent textures.

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n step1x-3d python=3.10
conda activate step1x-3d

# Install core dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

# Install custom components
cd step1x3d_texture/custom_rasterizer && python setup.py install
cd ../differentiable_renderer && python setup.py install
cd ../../
```

### Training Commands
```bash
# Train geometry VAE (autoencoder)
python train.py --config configs/train-geometry-autoencoder/michelangelo.yaml --train --gpu 0

# Train geometry diffusion model
python train.py --config configs/train-geometry-diffusion/step1x-3d-geometry-1300m.yaml --train --gpu 0

# Train with LoRA fine-tuning
python train.py --config configs/train-geometry-diffusion/step1x-3d-geometry-label-1300m.yaml --train --gpu 0 system.use_lora=True

# Train texture synthesis (multi-view generation)
python train_ig2mv.py --config configs/train-texture-ig2mv/step1x3d_ig2mv_sdxl.yaml --train

# Multi-GPU training example
torchrun train.py --config $config --train --gpu 0,1,2,3,4,5,6,7 trainer.num_nodes=$num_nodes
```

### Inference Commands
```bash
# Full pipeline inference
python inference.py

# Interactive demo
python app.py

# Data preprocessing
python data/watertight_and_sampling.py --input_mesh input.obj --skip_watertight
```

## Architecture Overview

### Two-Stage Pipeline
1. **Geometry Generation** (`step1x3d_geometry/`): Creates 3D meshes from single images using VAE-DiT architecture
2. **Texture Synthesis** (`step1x3d_texture/`): Applies multi-view consistent textures using SD-XL

### Key Components

#### Geometry Pipeline (`step1x3d_geometry/`)
- **Autoencoders** (`models/autoencoders/`): Michelangelo VAE for 3D shape compression
- **Conditional Encoders** (`models/conditional_encoders/`): DINOv2, T5, and label encoders for conditioning
- **Transformers** (`models/transformers/`): Flux-based transformer for denoising
- **Systems** (`systems/`): Training systems for autoencoder, diffusion, and rectified flow

#### Texture Pipeline (`step1x3d_texture/`)
- **Pipelines** (`pipelines/`): IG2MV SDXL pipeline for multi-view texture generation
- **Custom Rasterizer** (`custom_rasterizer/`): Hardware-accelerated rendering
- **Differentiable Renderer** (`differentiable_renderer/`): Backpropagation-compatible mesh rendering
- **Texture Sync** (`texture_sync/`): Cross-view consistency enforcement

### Model Variants
- **Base Geometry**: Image-to-3D generation (1.3B parameters)
- **Label Geometry**: Controllable generation with symmetry/edge control
- **Texture Synthesis**: Multi-view consistent texturing (3.5B parameters)

### Data Flow
1. Input image → Visual encoder → Conditional features
2. Random noise + conditions → Transformer → Denoised latents → VAE decoder → 3D geometry
3. Geometry + input image → Multi-view rendering → Texture synthesis → Textured 3D asset

## Configuration Files

### Training Configurations (`configs/`)
- **train-geometry-autoencoder/michelangelo.yaml**: Shape VAE training
- **train-geometry-diffusion/step1x-3d-geometry-1300m.yaml**: Base geometry diffusion
- **train-geometry-diffusion/step1x-3d-geometry-label-1300m.yaml**: Label-controlled geometry
- **train-texture-ig2mv/step1x3d_ig2mv_sdxl.yaml**: Multi-view texture synthesis

### Key Configuration Parameters
- `octree_resolution`: Resolution for 3D representations (default: 384)
- `max_facenum`: Maximum faces in generated mesh (default: 400000)
- `guidance_scale`: Classifier-free guidance strength (default: 7.5)
- `num_inference_steps`: Denoising steps (default: 50)

## Data Processing

### Preprocessing Pipeline
- **Watertight Processing**: Uses depth testing and winding numbers for robust mesh conversion
- **Sharp Edge Sampling**: Preserves geometric details during VAE encoding
- **Multi-view Rendering**: Generates 6 orthogonal views for texture training
- **Point Cloud Sampling**: FPS (Farthest Point Sampling) for geometry representation

### Data Formats
- **Input**: PNG/JPG images, OBJ/GLB meshes
- **Output**: GLB meshes with embedded textures
- **Intermediate**: TSDF/SDF representations, point clouds with normals

## Memory Requirements

| Model Configuration | GPU Memory | Inference Time |
|---------------------|------------|----------------|
| Geometry + Texture | 27GB | 152 seconds |
| Geometry + Texture (Label) | 29GB | 152 seconds |

## Common Development Patterns

### Pipeline Usage
```python
# Geometry generation
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
pipeline = Step1X3DGeometryPipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m')

# Texture synthesis
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline
texture_pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")
```

### Training Systems
- Extend `BaseSystem` for custom training logic
- Use `step1x3d_geometry.systems` for geometry-related training
- Use `step1x3d_texture.systems` for texture-related training

### Data Loading
- `step1x3d_geometry.data.Objaverse` for geometry datasets
- `step1x3d_texture.data.multiview` for texture datasets
- Custom data loaders should follow the established patterns

## Model Checkpoints

Models are distributed through HuggingFace Hub:
- `stepfun-ai/Step1X-3D/Step1X-3D-Geometry-1300m`
- `stepfun-ai/Step1X-3D/Step1X-3D-Geometry-Label-1300m`
- `stepfun-ai/Step1X-3D/Step1X-3D-Texture`

## Key Dependencies

- **PyTorch**: 2.5.1 with CUDA 12.4 support
- **PyTorch3D**: 3D deep learning operations
- **Kaolin**: NVIDIA's 3D deep learning library
- **Diffusers**: HuggingFace diffusion models
- **Trimesh**: Mesh processing utilities
- **OpenCV**: Image processing
- **PyTorch Lightning**: Training framework