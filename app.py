import os
import time
import uuid
import torch
import trimesh
import argparse
import numpy as np
import gradio as gr
import gc
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face


class MultiGPUDemoManager:
    """
    Manager for multi-GPU demo with persistent model loading
    """
    def __init__(self, geometry_model, texture_model, geometry_device="cuda:1", texture_device="cuda:0"):
        self.geometry_device = geometry_device
        self.texture_device = texture_device
        self.geometry_model_name = geometry_model
        self.texture_model_name = texture_model
        self.geometry_pipeline = None
        self.texture_pipeline = None
        
        # Check GPU availability
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Multi-GPU mode requires at least 2 GPUs")
        
        self._load_models()
    
    def _load_models(self):
        """Load all models and keep them in memory"""
        print(f"Loading geometry model on {self.geometry_device}")
        self.geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained(
            "stepfun-ai/Step1X-3D", subfolder=self.geometry_model_name
        ).to(self.geometry_device)
        
        print(f"Loading texture model on {self.texture_device}")
        self.texture_pipeline = Step1X3DTexturePipeline.from_pretrained(
            "stepfun-ai/Step1X-3D", subfolder=self.texture_model_name
        ).to(self.texture_device)
    
    def generate(self, input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type):
        """Generate using persistent models"""
        # Generate geometry
        if "Label" in self.geometry_model_name:
            out = self.geometry_pipeline(
                input_image_path,
                label={"symmetry": symmetry, "edge_type": edge_type},
                guidance_scale=float(guidance_scale),
                octree_resolution=384,
                max_facenum=int(max_facenum),
                num_inference_steps=int(inference_steps),
            )
        else:
            out = self.geometry_pipeline(
                input_image_path,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(inference_steps),
                max_facenum=int(max_facenum),
            )
        
        save_name = str(uuid.uuid4())
        geometry_save_path = f"{args.cache_dir}/{save_name}.glb"
        geometry_mesh = out.mesh[0]
        geometry_mesh.export(geometry_save_path)
        
        # Clean up intermediate results, keep models loaded
        del out
        torch.cuda.empty_cache()
        gc.collect()
        
        # Generate texture
        geometry_mesh = remove_degenerate_face(geometry_mesh)
        geometry_mesh = reduce_face(geometry_mesh)
        textured_mesh = self.texture_pipeline(input_image_path, geometry_mesh)
        textured_save_path = f"{args.cache_dir}/{save_name}-textured.glb"
        textured_mesh.export(textured_save_path)
        
        # Clean up intermediate results, keep models loaded
        del textured_mesh
        torch.cuda.empty_cache()
        gc.collect()
        
        return geometry_save_path, textured_save_path
    
    def cleanup(self):
        """Clean up all loaded models"""
        del self.geometry_pipeline, self.texture_pipeline
        torch.cuda.empty_cache()
        gc.collect()


# Global manager instance
gpu_manager = None


def generate_func(
    input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type
):
    global gpu_manager
    
    if gpu_manager is not None:
        # Multi-GPU mode
        print("Using Multi-GPU mode for generation")
        return gpu_manager.generate(
            input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type
        )
    else:
        # Single-GPU mode (original behavior)
        print("Using Single-GPU mode for generation")
        # Load geometry model
        geometry_model = Step1X3DGeometryPipeline.from_pretrained(
            "stepfun-ai/Step1X-3D", subfolder=args.geometry_model
        ).to("cuda")
        
        if "Label" in args.geometry_model:
            out = geometry_model(
                input_image_path,
                label={"symmetry": symmetry, "edge_type": edge_type},
                guidance_scale=float(guidance_scale),
                octree_resolution=384,
                max_facenum=int(max_facenum),
                num_inference_steps=int(inference_steps),
            )
        else:
            out = geometry_model(
                input_image_path,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(inference_steps),
                max_facenum=int(max_facenum),
            )

        save_name = str(uuid.uuid4())
        print(save_name)
        geometry_save_path = f"{args.cache_dir}/{save_name}.glb"
        geometry_mesh = out.mesh[0]
        geometry_mesh.export(geometry_save_path)

        # Clean up geometry model from VRAM
        del geometry_model, out
        torch.cuda.empty_cache()
        gc.collect()

        # Load texture model
        texture_model = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder=args.texture_model)
        
        geometry_mesh = remove_degenerate_face(geometry_mesh)
        geometry_mesh = reduce_face(geometry_mesh)
        textured_mesh = texture_model(input_image_path, geometry_mesh)
        textured_save_path = f"{args.cache_dir}/{save_name}-textured.glb"
        textured_mesh.export(textured_save_path)

        # Clean up texture model from VRAM
        del texture_model, textured_mesh
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Generate finish")
        return geometry_save_path, textured_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry_model", type=str, default="Step1X-3D-Geometry-Label-1300m"
    )
    parser.add_argument(
        "--texture_model", type=str, default="Step1X-3D-Texture"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["singlegpu", "multigpu"], 
        default="singlegpu",
        help="GPU mode: singlegpu (sequential) or multigpu (parallel)"
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Initialize multi-GPU manager if requested
    if args.mode == "multigpu":
        print("Initializing Multi-GPU Demo Manager...")
        gpu_manager = MultiGPUDemoManager(
            geometry_model=args.geometry_model,
            texture_model=args.texture_model
        )
        print("Multi-GPU Demo Manager initialized successfully!")
    else:
        print("Running in Single-GPU mode")

    with gr.Blocks(title="Step1X-3D demo") as demo:
        gr.Markdown("# Step1X-3D")
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(label="Image", type="filepath")
                guidance_scale = gr.Number(label="Guidance Scale", value="7.5")
                inference_steps = gr.Slider(
                    label="Inferece Steps", minimum=1, maximum=100, value=50
                )
                max_facenum = gr.Number(label="Max Face Num", value="400000")
                symmetry = gr.Radio(
                    choices=["x", "asymmetry"],
                    label="Symmetry Type",
                    value="x",
                    type="value",
                )
                edge_type = gr.Radio(
                    choices=["sharp", "normal", "smooth"],
                    label="Edge Type",
                    value="sharp",
                    type="value",
                )
                btn = gr.Button("Start")
            with gr.Column(scale=4):
                textured_preview = gr.Model3D(label="Textured", height=380)
                geometry_preview = gr.Model3D(label="Geometry", height=380)
            with gr.Column(scale=1):
                gr.Examples(
                    examples=[
                        ["examples/images/000.png"],
                        ["examples/images/001.png"],
                        ["examples/images/004.png"],
                        ["examples/images/008.png"],
                        ["examples/images/028.png"],
                        ["examples/images/032.png"],
                        ["examples/images/061.png"],
                        ["examples/images/107.png"],
                    ],
                    inputs=[input_image],
                    cache_examples=False,
                )

        btn.click(
            generate_func,
            inputs=[
                input_image,
                guidance_scale,
                inference_steps,
                max_facenum,
                symmetry,
                edge_type,
            ],
            outputs=[geometry_preview, textured_preview],
        )

    demo.launch(server_name=args.host, server_port=args.port)
    demo.queue(concurrency_count=3)
