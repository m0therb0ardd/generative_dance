# import cv2
# import numpy as np
# from ultralytics import YOLO

# model = YOLO("yolov8n-pose.pt")

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Define body-only joint indices (based on COCO format)
# # 0 = nose, 1 = left eye, 2 = right eye, 3 = left ear, 4 = right ear → skip these
# body_joint_indices = list(range(5, 17))  # Shoulders to ankles

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.predict(source=frame, show=False)

#     # Create a blank black canvas the same size as the frame
#     canvas = np.zeros_like(frame)

#     if results[0].keypoints is not None:
#         for kp in results[0].keypoints.xy:
#             for i in body_joint_indices:
#                 x, y = kp[i]
#                 cv2.circle(canvas, (int(x), int(y)), 6, (255, 255, 255), -1)

#     cv2.imshow("Pose Skeleton", canvas)

#     # Save the image when you press 's'
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         cv2.imwrite("pose.png", canvas)
#         print("Saved pose.png!")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# COCO-style joint connections
skeleton_pairs = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),         # left arm
    (6, 8), (8, 10),        # right arm
    (5, 11), (6, 12),       # shoulders to hips
    (11, 12),               # hips
    (11, 13), (13, 15),     # left leg
    (12, 14), (14, 16),     # right leg
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False)
    canvas = np.zeros_like(frame)

    if results[0].keypoints is not None:
        for kp in results[0].keypoints.xy:
            # Draw bones (lines)
            for i, j in skeleton_pairs:
                x1, y1 = kp[i]
                x2, y2 = kp[j]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:  # check validity
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

            # Draw joints (circles)
            for i in range(5, 17):  # skip face joints 0–4
                x, y = kp[i]
                if x > 0 and y > 0:
                    cv2.circle(canvas, (int(x), int(y)), 5, (255, 255, 255), -1)

    cv2.imshow("Live Pose Skeleton", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("pose_openpose.png", canvas)
        print("Saved pose_openpose.png!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

