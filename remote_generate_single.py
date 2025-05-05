import os
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

INPUT_PATH = os.path.expanduser("~/incoming_masks/input_test.png")
OUTPUT_PATH = os.path.expanduser("~/generated_waterfalls/output_test.png")

# === Load input image ===
if not os.path.exists(INPUT_PATH):
    print("❌ No input_test.png found!")
    exit()

input_image = Image.open(INPUT_PATH).convert("RGB").resize((768, 768))

# === Load model ===
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

# === Generate image ===
print(" Generating image...")
result = pipe(
    prompt="create mercury wave that follow the exact shape of the input, realistic",
    image=input_image,
    strength=0.95,
    num_inference_steps=4,
    guidance_scale=2.0
).images[0]

# === Save output ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result.save(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
