# === local_live_loop.py ===
import cv2
import numpy as np
import subprocess
import os
import time
from datetime import datetime
from ultralytics import YOLO


REMOTE_USER = "bsf0891"
REMOTE_IP = "129.105.69.10"
REMOTE_SESSION_ROOT = "~/incoming_sessions"
LOCAL_SESSION_ROOT = "sessions"

model = YOLO("yolov8n-seg.pt")
os.makedirs(LOCAL_SESSION_ROOT, exist_ok=True)

# Create timestamped session folder
session_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
local_session_path = os.path.join(LOCAL_SESSION_ROOT, session_timestamp)
os.makedirs(local_session_path, exist_ok=True)

image_index = 1
# cap = cv2.VideoCapture(0)
#change out of webcam to external
cap = cv2.VideoCapture(2)


print(f" Starting live capture into: {local_session_path}")

try:
    while True:
        print(f" Waiting for new pose...")

        start = time.time()
        while time.time() - start < 3:
            ret, preview_frame = cap.read()
            if not ret:
                print("Failed to read webcam")
                break

            # Show countdown on screen
            remaining = 3 - int(time.time() - start)
            frame_copy = preview_frame.copy()
            cv2.putText(frame_copy, f"Capture in {remaining}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            cv2.imshow(" Live Webcam (Press Q to Quit)", frame_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        # Warm up camera for fresh frame
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # YOLO segmentation
        results = model.predict(source=frame, show=False, conf=0.6, verbose=False)
        masks = results[0].masks

        if masks is None:
            print("⚠️ No mask detected, skipping.")
            continue

        # Process mask
        mask_img = (masks.data[0].cpu().numpy() * 255).astype("uint8")
        mask_img = cv2.merge([mask_img, mask_img, mask_img])
        inverted = cv2.bitwise_not(mask_img)
        dilated = cv2.dilate(inverted, np.ones((15, 15), np.uint8), iterations=1)
        cleaned = cv2.GaussianBlur(dilated, (11, 11), 0)

        # Save and send input
        input_filename = f"input{image_index}.png"
        local_input_path = os.path.join(local_session_path, input_filename)
        cv2.imwrite(local_input_path, cleaned)
        print(f"Saved {input_filename}")

        # Flash capture on live feed
        flash = preview_frame.copy()
        flash[:, :, 1] = 255  # green flash
        cv2.imshow(" Live Webcam (Pose Captured)", flash)
        cv2.waitKey(200)  # flash for 0.2 seconds

        # Save input image
        input_filename = f"input{image_index}.png"
        local_input_path = os.path.join(local_session_path, input_filename)
        cv2.imwrite(local_input_path, cleaned)
        print(f"Saved {input_filename}")

        # SCP input image
        remote_session_path = f"{REMOTE_SESSION_ROOT}/{session_timestamp}"
        subprocess.run(["ssh", f"{REMOTE_USER}@{REMOTE_IP}", f"mkdir -p {remote_session_path}"])
        subprocess.run(["scp", local_input_path, f"{REMOTE_USER}@{REMOTE_IP}:{remote_session_path}/"])
        print(f"Sent {input_filename} to lamb")

        # Wait for output
        output_filename = f"output{image_index}.png"
        local_output_path = os.path.join(local_session_path, output_filename)
        remote_output_path = f"{remote_session_path}/{output_filename}"

        while not os.path.exists(local_output_path):
            subprocess.run(["scp", f"{REMOTE_USER}@{REMOTE_IP}:{remote_output_path}", local_output_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)

        print(f"Received {output_filename}")

        # # Display side-by-side
        # input_img = cv2.imread(local_input_path)
        # output_img = cv2.imread(local_output_path)
        # if input_img is not None and output_img is not None:
        #     # Resize both images to same height (e.g., 768px)
        #     target_height = 768
        #     input_img = cv2.resize(input_img, (int(input_img.shape[1] * target_height / input_img.shape[0]), target_height))
        #     output_img = cv2.resize(output_img, (int(output_img.shape[1] * target_height / output_img.shape[0]), target_height))

        #     # Combine side by side
        #     print(f"input_img shape: {input_img.shape}, output_img shape: {output_img.shape}")
        #     print(f"input path exists: {os.path.exists(local_input_path)}, output path exists: {os.path.exists(local_output_path)}")
        #     side_by_side = np.hstack((input_img, output_img))

        #     cv2.imshow("Input ↔ Output", side_by_side)


        
        ################################################
        # Read both images
        input_img = cv2.imread(local_input_path)
        output_img = cv2.imread(local_output_path)

        # print(f"input_img shape: {input_img.shape}, output_img shape: {output_img.shape}")

        # if input_img is not None and output_img is not None:
        #     # Resize both to the same height
        #     target_height = 768
        #     input_img = cv2.resize(input_img, (int(input_img.shape[1] * target_height / input_img.shape[0]), target_height))
        #     output_img = cv2.resize(output_img, (int(output_img.shape[1] * target_height / output_img.shape[0]), target_height))

        #     # Display individually
        #     cv2.imshow("Input Only", input_img)
        #     cv2.waitKey(1000)

        #     cv2.imshow("Output Only", output_img)
        #     cv2.waitKey(1000)

        #     # Combine and display
        #     side_by_side = np.hstack((input_img, output_img))
        #     cv2.imshow("Combined View", side_by_side)
        #     cv2.waitKey(1000)

        #     # Optional: save combined image for visual check
        #     debug_filename = os.path.join(local_session_path, f"combined{image_index}.png")
        #     cv2.imwrite(debug_filename, side_by_side)
        #     print(f" Saved debug image: {debug_filename}")



        # Display only the output image in full screen
        cv2.namedWindow("Output Projection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output Projection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output Projection", output_img)
        # cv2.waitKey(1000)  # Show for 1 second or adjust as needed


        image_index += 1
        if cv2.waitKey(1000) == ord('q'):
            break

except KeyboardInterrupt:
    print("👋 Exiting loop")
finally:
    cap.release()
    cv2.destroyAllWindows()
