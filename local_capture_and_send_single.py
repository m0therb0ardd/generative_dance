import cv2
import numpy as np
import subprocess
import os
from ultralytics import YOLO

# SETTINGS
REMOTE_USERNAME = "bsf0891"
REMOTE_IP = "129.105.69.10"
REMOTE_DEST = "~/incoming_masks/input_test.png"
LOCAL_FILENAME = "input_test.png"

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Capture a single frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture frame from webcam.")
    exit()

# Run segmentation
results = model.predict(source=frame, show=False, conf=0.6, verbose=False)
masks = results[0].masks

if masks is not None:
    # Convert first mask to image
    mask_img = masks.data[0].cpu().numpy()
    mask_img = (mask_img * 255).astype("uint8")
    mask_img = cv2.merge([mask_img, mask_img, mask_img])  # convert to 3 channels

    # Invert, dilate, blur
    inverted = cv2.bitwise_not(mask_img)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    cleaned = cv2.GaussianBlur(dilated, (11, 11), 0)

    # Save and send
    cv2.imwrite(LOCAL_FILENAME, cleaned)
    print(f" Saved mask: {LOCAL_FILENAME}")

    subprocess.run([
        "scp", LOCAL_FILENAME,
        f"{REMOTE_USERNAME}@{REMOTE_IP}:{REMOTE_DEST}"
    ])
    print("Mask image sent to lamb.")
else:
    print(" No masks detected.")
