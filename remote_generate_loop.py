# === remote_generate_loop.py ===
import os
import time
import subprocess
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

REMOTE_USER = "catherine-maglione"
LOCAL_IP = "10.105.118.120"  # replace with your actual IP if it changes
LOCAL_SESSIONS_DIR = "~/gen_dance/sessions"
INCOMING_SESSIONS_DIR = os.path.expanduser("~/incoming_sessions")
os.makedirs(INCOMING_SESSIONS_DIR, exist_ok=True)
GENERATED_SESSIONS_DIR = os.path.expanduser("~/generated_waterfalls")

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

print("Watching for new input images...")

seen_folders = set()

while True:
    sessions = [f for f in os.listdir(INCOMING_SESSIONS_DIR) if os.path.isdir(os.path.join(INCOMING_SESSIONS_DIR, f))]
    sessions.sort()

    for session in sessions:
        in_path = os.path.join(INCOMING_SESSIONS_DIR, session)
        out_path = os.path.join(GENERATED_SESSIONS_DIR, session)
        os.makedirs(out_path, exist_ok=True)

        filenames = sorted([f for f in os.listdir(in_path) if f.startswith("input") and f.endswith(".png")])

        for fname in filenames:
            input_path = os.path.join(in_path, fname)
            output_path = os.path.join(out_path, fname.replace("input", "output"))

            if os.path.exists(output_path):
                continue  # already processed

            print(f"Generating from {fname}...")
            try:
                img = Image.open(input_path).convert("RGB").resize((768, 768))

                result = pipe(
                    prompt= "create an abstract mercury wave that follow the shape of the input, realistic",
                    # prompt = "create pod of whales that follow the exact shape of the input, realistic",
                    image=img,
                    strength=0.95,
                    num_inference_steps=4,
                    guidance_scale=2.0
                ).images[0]

                result.save(output_path)
                print(f"Saved: {output_path}")

                # SCP back to local
                remote_path = output_path
                local_path = f"{LOCAL_SESSIONS_DIR}/{session}/"
                subprocess.run([
                    "scp", remote_path,
                    f"{REMOTE_USER}@{LOCAL_IP}:{local_path}"
                ])
                print(f"⬆️  Sent back to local: {session}/{os.path.basename(output_path)}")

            except Exception as e:
                print(f"Error on {fname}: {e}")

    time.sleep(2)
