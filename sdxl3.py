# import torch
# from diffusers import StableDiffusionXLPipeline

# # Load SDXL Turbo
# pipeline = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# )
# pipeline = pipeline.to("cuda")  # or "cpu" if no GPU

# # Create an image
# prompt = "a waterfall shaped like a human body, cinematic, realistic, ethereal"
# image = pipeline(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

# # Save image
# image.save("output_waterfall.png")

#### 
# from diffusers import AutoPipelineForImage2Image
# from PIL import Image
# import torch
# import os

# # Set up pipeline
# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# )
# pipe.to("cuda")  # or "cpu" if you’re not using GPU

# # Load the input pose image
# init_image = Image.open("pose.png").convert("RGB").resize((768, 768))

# # List of prompts to test
# prompts = [
#     "A realistic waterfall scene, waterfall shape is the exact shape of a dancer's pose, realistic",
#     "Flowing water forming the silhouette of a human dancer, arms raised, water splash effect, cinematic lighting",
#     "A dancer made of cascading water, watery limbs, elegant shape, ethereal, glowing mist, blue tones",
#     "A liquid waterfall forming a ballet pose, flowing water mimics human body, dramatic contrast, realistic",
#     "Human pose outlined by a waterfall, graceful fluid arcs, transparent body, dark void background, soft mist"
# ]

# # Create output folder if needed
# os.makedirs("outputs", exist_ok=True)

# # Run batch generation
# for i, prompt in enumerate(prompts):
#     print(f"\n Generating image {i+1}/{len(prompts)}")
#     image = pipe(
#         prompt=prompt,
#         image=init_image,
#         strength=0.9,
#         num_inference_steps=4,
#         guidance_scale=5.0
#     ).images[0]

#     filename = f"outputs/waterfall_{i+1:03}.png"
#     image.save(filename)
#     print(f"Saved to {filename}")



# from diffusers import AutoPipelineForImage2Image
# from PIL import Image
# import torch
# import os

# # Set up pipeline
# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# )
# pipe.to("cuda")  # or "cpu" if no GPU

# # Load the input pose image
# init_image = Image.open("pose.png").convert("RGB").resize((768, 768))

# # Your *one* strong prompt
# prompt = "create a stream of water that follows the shape of the input, forest waterfall scene, realistic, cinematic, flowing water, misty"

# # Create output folder
# os.makedirs("outputs", exist_ok=True)

# # Strength and guidance values to test
# strength_values = [0.6, 0.7, 0.8, 0.9]
# guidance_values = [2.0, 4.0, 6.0]

# # Run batch
# counter = 1
# for strength in strength_values:
#     for guidance in guidance_values:
#         print(f"\n🌀 Generating: strength={strength}, guidance={guidance}")
#         image = pipe(
#             prompt=prompt,
#             image=init_image,
#             strength=strength,
#             num_inference_steps=4,
#             guidance_scale=guidance
#         ).images[0]

#         filename = f"outputs/waterfall_s{int(strength*10)}_g{int(guidance*10)}.png"
#         image.save(filename)
#         print(f"✅ Saved {filename}")



# from diffusers import AutoPipelineForImage2Image
# from PIL import Image
# import torch
# import os

# # Set up pipeline
# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# )
# pipe.to("cuda")  # or "cpu" if needed

# # Load the input pose image
# init_image = Image.open("pose.png").convert("RGB").resize((768, 768))

# # Strong base prompt
# #prompt = "create a stream of water that follows the shape of the input, fill in the rest of the background with nature,  waterfall scene, oil painting "
# # prompt = "create a pod of whales that follow the shape of the input, oil painting "
# # prompt = "create trash that follow the shape of the input, oil painting "
# prompt = "create water through rocks in forest that follow the shape of the input, photo realistic, misty, moody, twilight "



# # Create a **new outputs folder** for this fine batch
# os.makedirs("outputs_fine", exist_ok=True)

# # Fine-tuning small variations
# strength_values = [0.85, 0.9, 0.95]
# guidance_values = [1.8, 2.0, 2.2]

# # Run batch
# counter = 1
# for strength in strength_values:
#     for guidance in guidance_values:
#         print(f"\n Generating: strength={strength}, guidance={guidance}")
#         image = pipe(
#             prompt=prompt,
#             image=init_image,
#             strength=strength,
#             num_inference_steps=4,
#             guidance_scale=guidance
#         ).images[0]

#         filename = f"outputs_fine/whale{int(strength*10)}_g{int(guidance*10)}.png"
#         image.save(filename)
#         print(f"Saved {filename}")



from diffusers import AutoPipelineForImage2Image
from PIL import Image
import torch

# Load SDXL Turbo img2img pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

# Load your captured pose image
pose_image = Image.open("captured_body_mask.png").convert("RGB").resize((768, 768))

# Run image-to-image
prompt = "create water through rocks in forest that follow the shape of the input, photo realistic, misty, moody, twilight "
result = pipe(
    prompt=prompt,
    image=pose_image,
    strength=0.8,    # YOUR discovered sweet spot!
    num_inference_steps=4,
    guidance_scale=2.0
).images[0]

# Save the generated waterfall dancer
result.save("generated_waterfall.png")
