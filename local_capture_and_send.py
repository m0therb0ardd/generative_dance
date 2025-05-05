import cv2
import numpy as np
import subprocess
import os
import datetime
import time
from ultralytics import YOLO

# Settings
REMOTE_USERNAME = "bsf0891"
REMOTE_IP = "129.105.69.10"
REMOTE_FOLDER = "~/incoming_masks/"
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
LOCAL_INPUT_FOLDER = f"input_{SESSION_TIMESTAMP}/"
LOCAL_OUTPUT_FOLDER = f"output_{SESSION_TIMESTAMP}/"
os.makedirs(LOCAL_INPUT_FOLDER, exist_ok=True)
os.makedirs(LOCAL_OUTPUT_FOLDER, exist_ok=True)

# Load model and open webcam
model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(0)

frame_counter = 0
process_frame_interval = 5
last_capture_time = time.time() - 10
display_img = None
img_index = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % process_frame_interval == 0:
        results = model.predict(source=frame, show=False, conf=0.6, verbose=False)
        masks = results[0].masks

        if masks is not None:
            mask_img = masks.data[0].cpu().numpy()
            mask_img = (mask_img * 255).astype("uint8")
            mask_img = cv2.merge([mask_img, mask_img, mask_img])
            inverted_mask = cv2.bitwise_not(mask_img)
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(inverted_mask, kernel, iterations=1)
            cleaned = cv2.GaussianBlur(dilated, (11, 11), 0)
            display_img = cleaned

    if time.time() - last_capture_time >= 10 and display_img is not None:
        pose_filename = f"input{img_index}.png"
        pose_path = os.path.join(LOCAL_INPUT_FOLDER, pose_filename)
        cv2.imwrite(pose_path, display_img)
        print(f"✅ Saved: {pose_path}")

        subprocess.run([
            "scp", pose_path,
            f"{REMOTE_USERNAME}@{REMOTE_IP}:{REMOTE_FOLDER}"
        ])
        print(f"🚀 Sent {pose_filename} to lamb")

        expected_waterfall = f"output{img_index}.png"
        remote_output_path = f"~/generated_waterfalls/{expected_waterfall}"
        local_output_path = os.path.join(LOCAL_OUTPUT_FOLDER, expected_waterfall)

        # Wait + pull result
        while not os.path.exists(local_output_path):
            subprocess.run([
                "scp", f"{REMOTE_USERNAME}@{REMOTE_IP}:{remote_output_path}", LOCAL_OUTPUT_FOLDER
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)

        img = cv2.imread(local_output_path)
        if img is not None:
            cv2.imshow('Generated Waterfall', img)

        img_index += 1
        last_capture_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
