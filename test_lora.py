#!/usr/bin/env python3
"""Quick script to test your LoRA and generate sample images."""

import torch
from diffusers import FluxPipeline

# Configuration
LORA_PATH = "./output/my_character_lora.safetensors"
BASE_MODEL = "/home/azureuser/flux.2-dev"
PROMPT = "your concept here, high quality photo"  # Customize this
OUTPUT_PATH = "./output/test_sample.png"

print("Loading FLUX2-dev model...")
pipe = FluxPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

print(f"Loading LoRA from {LORA_PATH}...")
pipe.load_lora_weights(".", weight_name=LORA_PATH.replace("./", ""))

print(f"Generating image with prompt: {PROMPT}")
image = pipe(
    prompt=PROMPT,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024,
    width=1024,
).images[0]

image.save(OUTPUT_PATH)
print(f"✓ Image saved to {OUTPUT_PATH}")
print("\nTry different prompts to test your LoRA!")
