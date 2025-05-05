# import os
# import time
# import subprocess
# import datetime
# from PIL import Image
# import torch
# from diffusers import AutoPipelineForImage2Image

# # SETTINGS
# LOCAL_USERNAME = "catherine-maglione"
# LOCAL_IP = "10.105.118.120"
# LOCAL_FOLDER = "/home/catherine-maglione/gen_dance/waterfall_output/"
# INCOMING_FOLDER = os.path.expanduser("~/incoming_masks/")
# GENERATED_FOLDER = os.path.expanduser("~/generated_waterfalls/")

# os.makedirs(GENERATED_FOLDER, exist_ok=True)

# # Load SDXL-Turbo
# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# )
# pipe.to("cuda")

# print("Looking for new pose masks...")

# already_processed = set()

# while True:
#     files = os.listdir(INCOMING_FOLDER)
#     png_files = [f for f in files if f.endswith(".png")]

#     for filename in png_files:
#         filepath = os.path.join(INCOMING_FOLDER, filename)

#         if filename not in already_processed:
#             print(f"Found new mask: {filename}")
#             try:
#                 pose_image = Image.open(filepath).convert("RGB").resize((768, 768))
#                 # prompt = "create water through rocks in forest that follow the shape of the input, photo realistic, misty, moody, twilight "
#                 # prompt = "create whales in the input, photo realistic, misty, moody "
#                 prompt = "create waves that follow the exact shape in the input, photo realistic, misty, moody "


#                 result = pipe(
#                     prompt=prompt,
#                     image=pose_image,
#                     strength=0.8,
#                     num_inference_steps=4,
#                     guidance_scale=2.0
#                 ).images[0]

#                 timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#                 out_filename = f"waterfall_{timestamp}.png"
#                 out_filepath = os.path.join(GENERATED_FOLDER, out_filename)
#                 result.save(out_filepath)

#                 print(f"✅ Generated waterfall: {out_filename}")

#                 # SCP back to local machine
#                 scp_command = [
#                     "scp",
#                     out_filepath,
#                     f"{LOCAL_USERNAME}@{LOCAL_IP}:{LOCAL_FOLDER}"
#                 ]
#                 subprocess.run(scp_command)
#                 print(f"Sent {out_filename} back to laptop!")

#                 already_processed.add(filename)

#             except Exception as e:
#                 print(f" Error processing {filename}: {e}")

#     time.sleep(2)  # Check for new files every 2 seconds\

import os
import time
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

# SESSION folder name (match the one from local)
SESSION_FOLDER = "input_2025-05-02-10-32-30"  # 🟡 You update this on each session
INPUT_FOLDER = os.path.expanduser(f"~/incoming_masks/{SESSION_FOLDER}")
OUTPUT_FOLDER = os.path.expanduser(f"~/generated_waterfalls/{SESSION_FOLDER}")

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

print(f"🎯 Watching folder: {SESSION_FOLDER}")

PROCESSED = set()
img_index = 1

while True:
    files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.startswith("input") and f.endswith(".png")],
        key=lambda x: os.path.getmtime(os.path.join(INPUT_FOLDER, x))
    )

    for f in files:
        if f in PROCESSED:
            continue

        try:
            img_path = os.path.join(INPUT_FOLDER, f)
            input_img = Image.open(img_path).convert("RGB").resize((768, 768))

            print(f"Generating from {f}...")
            output_img = pipe(
                prompt="create waves that follow the exact shape in the input, photo realistic, misty, moody",
                image=input_img,
                strength=0.95,
                num_inference_steps=4,
                guidance_scale=2.0
            ).images[0]

            output_name = f"output{img_index}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            output_img.save(output_path)

            print(f"Saved: {output_name}")
            PROCESSED.add(f)
            img_index += 1

        except Exception as e:
            print(f"❌ Failed on {f}: {e}")

    time.sleep(2)

