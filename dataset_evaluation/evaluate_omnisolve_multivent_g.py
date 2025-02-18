from google import genai
from dotenv import load_dotenv
import os
from PIL import Image
from google.genai import types
from io import BytesIO


load_dotenv()


import json
import random
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import cv2

# Example detection dictionary
detection = {
    "role": "what",
    "entity": "a part of a text banner saying \"floods\"",
    "frame": 58,
    "bbox": [62.421875, 77.833333, 79.015625, 88.333333],  # [x1, y1, x2, y2]
    "certainty": 5,
    "ocr_flag": True
}

detection = {
    "mask_name": "/data/multivent_processed_without_delay/0fXN4hBjQQg/0_detection_results/mask_data/mask_00167.npy", 
    "mask_height": 1280, 
    "mask_width": 720, 
    "promote_type": "mask", 
    "labels": {
        "8": {
            "instance_id": 8, 
            "class_name": "tree", 
            "x1": 0, 
            "y1": 0, 
            "x2": 0, 
            "y2": 0, 
            "logit": 0.0
            }, 
        "9": {
            "instance_id": 9,
            "class_name": "car", 
            "x1": 0, 
            "y1": 0, 
            "x2": 0, 
            "y2": 0, 
            "logit": 0.0
            }, 
        "10": {
            "instance_id": 10,
            "class_name": "flood road", 
            "x1": 0, 
            "y1": 789, 
            "x2": 719, 
            "y2": 1279, 
            "logit": 0.0
            }, 
        "11": {"instance_id": 11, "class_name": "tree", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "logit": 0.0}, "12": {"instance_id": 12, "class_name": "flood", "x1": 0, "y1": 817, "x2": 719, "y2": 1279, "logit": 0.0}, "13": {"instance_id": 13, "class_name": "rain", "x1": 0, "y1": 4, "x2": 719, "y2": 1279, "logit": 0.0}, "6": {"instance_id": 6, "class_name": "fog", "x1": 0, "y1": 0, "x2": 719, "y2": 602, "logit": 0.0}, "14": {"instance_id": 14, "class_name": "tree", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "logit": 0.0}}}


detection = {"mask_name": "/data/multivent_processed_without_delay/0fXN4hBjQQg/0_detection_results/mask_data/mask_00031.npy", "mask_height": 1280, "mask_width": 720, "promote_type": "mask", "labels": {"1": {"instance_id": 1, "class_name": "sign", "x1": 325, "y1": 911, "x2": 579, "y2": 1031, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "tree", "x1": 107, "y1": 209, "x2": 497, "y2": 545, "logit": 0.0}, "3": {"instance_id": 3, "class_name": "tree", "x1": 0, "y1": 393, "x2": 132, "y2": 622, "logit": 0.0}, "4": {"instance_id": 4, "class_name": "road", "x1": 0, "y1": 553, "x2": 719, "y2": 1063, "logit": 0.0}, "5": {"instance_id": 5, "class_name": "flood", "x1": 0, "y1": 555, "x2": 719, "y2": 1063, "logit": 0.0}, "6": {"instance_id": 6, "class_name": "rain", "x1": 0, "y1": 0, "x2": 719, "y2": 1279, "logit": 0.0}, "7": {"instance_id": 7, "class_name": "tree", "x1": 401, "y1": 452, "x2": 587, "y2": 714, "logit": 0.0}}}

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_detection_on_frame(video_path, detection, frame_index=167):
    """
    Displays a single frame from the video with:
    1) Segmentation mask overlaid
    2) Bounding boxes drawn
    """

    # 1. Read the video frame
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Safety check: ensure frame_index is within the video
    if frame_index >= total_frames:
        print(f"Requested frame_index={frame_index} exceeds total frames={total_frames}.")
        cap.release()
        return

    # Jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame {frame_index} from {video_path}.")
        return

    # Convert BGR to RGB for consistent plotting with matplotlib
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 2. Load the segmentation mask (e.g. "mask_00167.npy")
    mask_path = detection["mask_name"]
    mask = np.load(mask_path)  # shape: (mask_height, mask_width)

    # Sanity check: if mask doesn't match frame size, you may need to resize
    mask_height, mask_width = mask.shape[:2]
    frame_height, frame_width = frame_rgb.shape[:2]
    if (mask_height != frame_height) or (mask_width != frame_width):
        print("Warning: Mask dimensions differ from frame dimensions. Resizing mask to match frame.")
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    # 3. Create a color map for each instance_id
    #    Let's assign a random color to each unique instance_id in the mask.
    unique_ids = np.unique(mask)
    np.random.seed(42)  # for reproducible colors
    id_to_color = {}
    for inst_id in unique_ids:
        if inst_id == 0:
            # zero is often background; give it a transparent color or skip it
            id_to_color[inst_id] = (0, 0, 0, 0)  # RGBA with alpha=0
        else:
            color = np.random.randint(0, 255, size=3)  # random BGR
            id_to_color[inst_id] = (*color, 180)       # (B, G, R, alpha)
    
    # 4. Overlay the colorized mask on the frame
    #    We'll create an RGBA overlay and then blend it onto the frame.
    overlay_rgba = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

    for inst_id in unique_ids:
        overlay_rgba[mask == inst_id] = id_to_color[inst_id]

    # Convert frame to RGBA
    frame_rgba = np.concatenate([frame_rgb, np.full((frame_height, frame_width, 1), 255, dtype=np.uint8)], axis=-1)

    # Alpha blend: new_frame = alpha*overlay + (1-alpha)*frame
    alpha_mask = overlay_rgba[..., 3] / 255.0
    for c in range(3):  # blend each color channel
        frame_rgba[..., c] = (frame_rgba[..., c] * (1 - alpha_mask) + 
                              overlay_rgba[..., c] * alpha_mask).astype(np.uint8)

    # 5. Draw bounding boxes
    labels = detection["labels"]
    
    # For display in matplotlib, we'll annotate on the RGBA image
    # Convert RGBA->RGB for drawing with matplotlib
    disp_img = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)

    for label_id, label_data in labels.items():
        class_name = label_data["class_name"]
        x1, y1, x2, y2 = label_data["x1"], label_data["y1"], label_data["x2"], label_data["y2"]
        
        # Skip degenerate boxes
        if x1 == x2 and y1 == y2:
            continue
        
        # Draw rectangle on disp_img
        color = (255, 0, 0)  # default color in BGR
        thickness = 2
        disp_img = cv2.rectangle(disp_img, (x1, y1), (x2, y2), color, thickness)
        
        # Put class text above the bounding box
        disp_img = cv2.putText(
            disp_img, class_name, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    # 6. Show the result using matplotlib
    plt.figure(figsize=(8, 12))
    plt.imshow(disp_img)
    plt.axis('off')
    plt.title(f"Frame {frame_index} Visualization")
    plt.show()


detection = {
    "role": "what",
    "entity": "a part of a text banner saying \"floods\"",
    "frame": 58,
    "bbox": [62.421875, 77.833333, 79.015625, 88.333333],  # [x1, y1, x2, y2]
    "certainty": 5,
    "ocr_flag": True
}

detection = {
    "mask_name": "/data/multivent_processed_without_delay/0fXN4hBjQQg/0_detection_results/mask_data/mask_00167.npy", 
    "mask_height": 1280, 
    "mask_width": 720, 
    "promote_type": "mask", 
    "labels": {
        "8": {
            "instance_id": 8, 
            "class_name": "tree", 
            "x1": 0, 
            "y1": 0, 
            "x2": 0, 
            "y2": 0, 
            "logit": 0.0
            }, 
        "9": {
            "instance_id": 9,
            "class_name": "car", 
            "x1": 0, 
            "y1": 0, 
            "x2": 0, 
            "y2": 0, 
            "logit": 0.0
            }, 
        "10": {
            "instance_id": 10,
            "class_name": "flood road", 
            "x1": 0, 
            "y1": 789, 
            "x2": 719, 
            "y2": 1279, 
            "logit": 0.0
            }, 
        "11": {"instance_id": 11, "class_name": "tree", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "logit": 0.0}, "12": {"instance_id": 12, "class_name": "flood", "x1": 0, "y1": 817, "x2": 719, "y2": 1279, "logit": 0.0}, "13": {"instance_id": 13, "class_name": "rain", "x1": 0, "y1": 4, "x2": 719, "y2": 1279, "logit": 0.0}, "6": {"instance_id": 6, "class_name": "fog", "x1": 0, "y1": 0, "x2": 719, "y2": 602, "logit": 0.0}, "14": {"instance_id": 14, "class_name": "tree", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "logit": 0.0}}}



def format_omnisolve_results_correctly(multivent_g_result_path, multivent_g_json_path, frame_convert_json_path, output_filepath):
    with open(multivent_g_json_path, "r") as f:
        gt = json.load(f)
    
    with open(frame_convert_json_path, "r") as f:
        frame_conversion = json.load(f)
    
    formatted_results = {}
    
    for video in tqdm(gt):
        video_folder = os.path.join(multivent_g_result_path, video)
        if os.path.exists(video_folder) and video in frame_conversion:
            video_objects = []
            for frame in frame_conversion[video]:
                
                subclip = frame_conversion[video][frame][0][-5]
                subclip_frame = frame_conversion[video][frame][1]
                subclip_name = str(subclip_frame)
                while len(subclip_name) != 5:
                    subclip_name = "0" + subclip_name
                subclip_name = os.path.join(video_folder, f"{subclip}_detection_results", "json_data", f"mask_{subclip_name}.json")
                
                if not os.path.exists(subclip_name):
                    continue
                
                with open(subclip_name, "r") as f:
                    subclip_frame_info = json.load(f)

                for object in subclip_frame_info["labels"]:
                    object_info = subclip_frame_info["labels"][object]

                    # # Normalize the bounding boxes
                    # video_path = os.path.join(video_folder, f"{subclip}.mp4")
                    # cap = cv2.VideoCapture(video_path)
                    # cap.set(cv2.CAP_PROP_POS_FRAMES, subclip_frame)
                    # ret, frame_bgr = cap.read()
                    # cap.release()
                    # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # frame_height, frame_width = frame_rgb.shape[:2]


                    new_object = {
                        "entity":object_info["class_name"],
                        "frame": int(frame),
                        "bbox": [
                            object_info["x1"] / subclip_frame_info["mask_width"] * 100,
                            object_info["y1"] / subclip_frame_info["mask_height"] * 100,
                            object_info["x2"] / subclip_frame_info["mask_width"] * 100,
                            object_info["y2"] / subclip_frame_info["mask_height"] * 100
                            ],
                        "subclip_frame": subclip_frame,
                        "subclip_location": subclip_name,
                        "mask_height": subclip_frame_info["mask_height"],
                        "mask_width": subclip_frame_info["mask_width"]
                    }
                    video_objects.append(new_object)
            formatted_results[video] = video_objects
    
        with open(output_filepath, "w") as f:
            json.dump(formatted_results, f, indent=2)




multivent_yt_path = "/data/multivent_yt_videos/"
multivent_g_result_path = "/data/multivent_processed_without_delay/"
multivent_g_json_path = "/home/aiden/Documents/cs/multiVENT/data/multivent_g.json"
frame_convert_json_path = os.path.join(multivent_g_result_path, "frame_convert.json")
output_filepath = "benchmark_results/omniverse_formatted.json"

format_omnisolve_results_correctly(multivent_g_result_path, multivent_g_json_path, frame_convert_json_path, output_filepath)
# visualize_detection_on_frame("/data/multivent_processed_without_delay/0fXN4hBjQQg/0.mp4", detection)






