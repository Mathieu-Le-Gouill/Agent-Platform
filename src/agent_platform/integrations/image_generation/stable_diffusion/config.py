from __future__ import annotations

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig


class StableDiffusionConfig(ImageGenConfig):
    # HuggingFace hub repo id loaded via StableDiffusionPipeline.from_pretrained.
    model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # Torch device the pipeline is moved to (e.g. "cpu", "cuda", "mps").
    device: str = "cpu"
    # Torch dtype used to load pipeline weights.
    dtype: str = "float32"
    # Whether to run the built-in NSFW safety checker.
    safety_checker: bool = True
    # Classifier-free guidance scale; higher follows the prompt more closely, at the cost of diversity/quality.
    guidance_scale: float = 7.5
    # Number of denoising steps; more steps generally trade runtime for image quality.
    num_inference_steps: int = 50
    # Prompt describing what to steer generation away from.
    negative_prompt: str | None = None
    # Seed for a torch.Generator(device=...).manual_seed(...) to make generation reproducible.
    seed: int | None = None


"""
sources: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
         https://huggingface.co/docs/diffusers/using-diffusers/loading
         https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds
"""
