import warnings

warnings.filterwarnings("ignore")
import os
import trimesh
import argparse
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)

from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import torch
import gc


class MultiGPUInferenceManager:
    """
    Manager for multi-GPU inference with persistent model loading
    """
    def __init__(self, geometry_device="cuda:1", texture_device="cuda:0"):
        self.geometry_device = geometry_device
        self.texture_device = texture_device
        self.geometry_pipeline = None
        self.geometry_label_pipeline = None
        self.texture_pipeline = None
        
        # Check GPU availability
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Multi-GPU mode requires at least 2 GPUs")
        
        self._load_models()
    
    def _load_models(self):
        """Load all models and keep them in memory"""
        print(f"Loading geometry models on {self.geometry_device}")
        self.geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained(
            "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
        ).to(self.geometry_device)
        
        self.geometry_label_pipeline = Step1X3DGeometryPipeline.from_pretrained(
            "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-Label-1300m'
        ).to(self.geometry_device)
        
        print(f"Loading texture model on {self.texture_device}")
        self.texture_pipeline = Step1X3DTexturePipeline.from_pretrained(
            "stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture"
        ).to(self.texture_device)
    
    def generate_geometry(self, input_image_path, save_glb_path, use_label=False, **kwargs):
        """Generate geometry using persistent models"""
        pipeline = self.geometry_label_pipeline if use_label else self.geometry_pipeline
        generator = torch.Generator(device=self.geometry_device)
        generator.manual_seed(2025)
        
        if use_label:
            out = pipeline(
                input_image_path,
                label=kwargs.get("label", {"symmetry": "x", "edge_type": "sharp"}),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                octree_resolution=kwargs.get("octree_resolution", 384),
                max_facenum=kwargs.get("max_facenum", 400000),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
                generator=generator
            )
        else:
            out = pipeline(
                input_image_path,
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
                generator=generator
            )
        
        os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
        out.mesh[0].export(save_glb_path)
        
        # Clean up intermediate results, keep models loaded
        del generator, out
        torch.cuda.empty_cache()
        gc.collect()
    
    def generate_texture(self, input_image_path, input_glb_path, save_glb_path):
        """Generate texture using persistent model"""
        mesh = trimesh.load(input_glb_path)
        mesh = remove_degenerate_face(mesh)
        mesh = reduce_face(mesh)
        
        # Generate texture on texture device
        textured_mesh = self.texture_pipeline(input_image_path, mesh, seed=2025)
        
        os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
        textured_mesh.export(save_glb_path)
        
        # Clean up intermediate results, keep models loaded
        del textured_mesh
        torch.cuda.empty_cache()
        gc.collect()
    
    def cleanup(self):
        """Clean up all loaded models"""
        del self.geometry_pipeline, self.geometry_label_pipeline, self.texture_pipeline
        torch.cuda.empty_cache()
        gc.collect()


def geometry_pipeline(input_image_path, save_glb_path):
    """
    The base geometry model, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
    ).to("cuda")

    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)
    out = pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50, generator=generator)

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)
    
    # Clean up VRAM after geometry generation
    del pipeline, generator, out
    torch.cuda.empty_cache()
    gc.collect()


def geometry_label_pipeline(input_image_path, save_glb_path):
    """
    The label geometry model, support using label to control generation, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-Label-1300m'
    ).to("cuda")
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)

    out = pipeline(
        input_image_path,
        label={"symmetry": "x", "edge_type": "sharp"},
        guidance_scale=7.5,
        octree_resolution=384,
        max_facenum=400000,
        num_inference_steps=50,
        generator=generator
    )

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)
    
    # Clean up VRAM after geometry generation
    del pipeline, generator, out
    torch.cuda.empty_cache()
    gc.collect()


def texture_pipeline(input_image_path, input_glb_path, save_glb_path):
    """
    The texture model, input image and glb generate textured glb
    """
    mesh = trimesh.load(input_glb_path)
    pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")
    mesh = remove_degenerate_face(mesh)
    mesh = reduce_face(mesh)
    textured_mesh = pipeline(input_image_path, mesh, seed=2025)
    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    textured_mesh.export(save_glb_path)
    
    # Clean up VRAM after texture generation
    del pipeline, textured_mesh
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step1X-3D Inference")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["singlegpu", "multigpu"], 
        default="singlegpu",
        help="GPU mode: singlegpu (sequential) or multigpu (parallel)"
    )
    parser.add_argument(
        "--input_image", 
        type=str, 
        default="examples/images/000.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Output directory for generated models"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "multigpu":
        print("Running in Multi-GPU mode...")
        manager = MultiGPUInferenceManager()
        
        # Generate base geometry
        base_glb = f"{args.output_dir}/000.glb"
        manager.generate_geometry(args.input_image, base_glb)
        
        # Generate label geometry
        label_glb = f"{args.output_dir}/000-label.glb"
        manager.generate_geometry(args.input_image, label_glb, use_label=True)
        
        # Generate textures
        manager.generate_texture(args.input_image, base_glb, f"{args.output_dir}/000-textured.glb")
        manager.generate_texture(args.input_image, label_glb, f"{args.output_dir}/000-label-textured.glb")
        
        manager.cleanup()
    else:
        print("Running in Single-GPU mode...")
        image_path = args.input_image
        geometry_pipeline(image_path, f"{args.output_dir}/000.glb")
        geometry_label_pipeline(image_path, f"{args.output_dir}/000-label.glb")
        texture_pipeline(image_path, f"{args.output_dir}/000.glb", f"{args.output_dir}/000-textured.glb")
        texture_pipeline(
            image_path, f"{args.output_dir}/000-label.glb", f"{args.output_dir}/000-label-textured.glb"
        )
