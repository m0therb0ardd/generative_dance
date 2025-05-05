import os
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

# === SETTINGS ===
INPUT_IMAGE_PATH = "input_test.png"  # ← make sure this exists (768x768 recommended)
OUTPUT_FOLDER = "batch_outputs"
PROMPT = "a wave that follows the exact shape of the input image, ethereal, misty, high detail"

# Define parameter grid
param_grid = [
    {"strength": 0.85, "guidance": 1.5},
    {"strength": 0.90, "guidance": 2.0},
    {"strength": 0.95, "guidance": 2.0},
    {"strength": 0.90, "guidance": 2.0},
    {"strength": 0.85, "guidance": 2.0},
]

# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load input image
input_image = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize((768, 768))

# Load model
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")  # Use "cpu" if needed

# Run variations
for i, params in enumerate(param_grid):
    strength = params["strength"]
    guidance = params["guidance"]

    print(f"🔁 Generating with strength={strength}, guidance={guidance}...")

    result = pipe(
        prompt=PROMPT,
        image=input_image,
        strength=strength,
        num_inference_steps=4,
        guidance_scale=guidance
    ).images[0]

    filename = f"output_s{int(strength*100)}_g{int(guidance)}.png"
    result.save(os.path.join(OUTPUT_FOLDER, filename))
    print(f"✅ Saved {filename}")

print("\n🎉 Batch complete! Check the 'batch_outputs/' folder.")
