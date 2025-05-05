import cv2
import os
import time
import numpy as np
from glob import glob

def get_latest_folder(prefix):
    folders = [f for f in os.listdir(".") if os.path.isdir(f) and f.startswith(prefix)]
    if not folders:
        return None
    folders.sort()  # assumes timestamp format
    return folders[-1]

def load_pair(input_folder, output_folder, index):
    input_path = os.path.join(input_folder, f"input{index}.png")
    output_path = os.path.join(output_folder, f"output{index}.png")

    if not os.path.exists(input_path) or not os.path.exists(output_path):
        return None, None

    input_img = cv2.imread(input_path)
    output_img = cv2.imread(output_path)

    if input_img is None or output_img is None:
        return None, None

    # Resize to same height
    height = 768
    input_img = cv2.resize(input_img, (int(input_img.shape[1] * height / input_img.shape[0]), height))
    output_img = cv2.resize(output_img, (int(output_img.shape[1] * height / output_img.shape[0]), height))

    # Pad to same width
    max_width = max(input_img.shape[1], output_img.shape[1])
    input_img = cv2.copyMakeBorder(input_img, 0, 0, 0, max_width - input_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    output_img = cv2.copyMakeBorder(output_img, 0, 0, 0, max_width - output_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return input_img, output_img

def main():
    input_folder = get_latest_folder("input_")
    output_folder = get_latest_folder("output_")

    if not input_folder or not output_folder:
        print("❌ Could not find input_*/output_* folders.")
        return

    print(f"🎥 Watching: {input_folder} + {output_folder}")

    cv2.namedWindow("Pose + Waterfall", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose + Waterfall", 1600, 768)

    index = 1
    while True:
        input_img, output_img = load_pair(input_folder, output_folder, index)

        if input_img is not None and output_img is not None:
            combined = np.hstack((input_img, output_img))
            cv2.imshow("Pose + Waterfall", combined)
            index += 1
        else:
            time.sleep(1)

        key = cv2.waitKey(1000)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
