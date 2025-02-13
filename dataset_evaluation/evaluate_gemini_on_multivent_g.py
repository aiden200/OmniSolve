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



def load_frame_as_pil(video_path, frame_number):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_number} from {video_path}")

    # OpenCV loads images in BGR by default; convert it to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Create a PIL Image from the Numpy array
    pil_image = Image.fromarray(frame_rgb)

    return pil_image


def load_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        cap.release()
        return None
    
    cap.release()
    return frame

def show_frame_with_detection(video_path, detection):
    """
    1. Loads the specified frame from video_path,
    2. Converts the bounding box from 'percent' to 'pixels',
    3. Draws the bounding box and label on the frame,
    4. Displays the result in an OpenCV window.
    """
    frame_idx = detection["frame"]
    bbox_perc = detection["bbox"]  # [x1%, y1%, x2%, y2%]
    label_text = detection["entity"]

    # 1. Load the specified frame
    frame = load_frame(video_path, frame_idx)
    if frame is None:
        return
    
    # 2. Convert bounding box from percent to actual pixel coordinates
    height, width = frame.shape[:2]  # (rows, cols)
    
    # unpack and scale
    x1p, y1p, x2p, y2p = bbox_perc
    x1 = int(x1p / 100.0 * width)
    y1 = int(y1p / 100.0 * height)
    x2 = int(x2p / 100.0 * width)
    y2 = int(y2p / 100.0 * height)
    
    # 3. Draw bounding box and label
    color = (0, 255, 0)   # green BGR
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Optional: put the label text right above the bounding box
    cv2.putText(frame, label_text, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 4. Display the frame
    cv2.imshow("Frame with Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]


def parse_gemini_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def convert_coords_back(bounding_boxes, frame_number):
    # width, height = im.size
    bounding_boxes = parse_gemini_json(bounding_boxes)

    objects = []

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
        
        # print(bounding_box)
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = float(bounding_box["bbox"][0])
        abs_x1 = float(bounding_box["bbox"][1])
        abs_y2 = float(bounding_box["bbox"][2])
        abs_x2 = float(bounding_box["bbox"][3])

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        object = {
           "entity": bounding_box["label"],
           "frame": frame_number,
           "bbox": [abs_x1, abs_y1, abs_x2, abs_y2]}
        objects.append(object)
    
    return objects
        




def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    print(bounding_boxes)

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_gemini_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox"][0]/1000 * height)
      abs_x1 = int(bounding_box["bbox"][1]/1000 * width)
      abs_y2 = int(bounding_box["bbox"][2]/1000 * height)
      abs_x2 = int(bounding_box["bbox"][3]/1000 * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()



def test_gemini(multivent_yt_path, multivent_g_results_path, multivent_g_json_path, results_file="benchmark_results/gemini_results.json"):

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))



    bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    The output json should follow the following format:
[
  {
    "label": [LABEL1],
    "bbox": [
        
    ]
  },
  {
    "label": [LABEL2],
    "bbox": [

    ]
  }, ...
]
      """


    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]


    prompt = "Detect the 2d bounding boxes of the object"  # @param {type:"string"}


    with open(multivent_g_json_path, "r") as f:
        multivent_g_ground_truth = json.load(f)

    videos = [name for name in os.listdir(multivent_g_results_path) if os.path.isdir(os.path.join(multivent_g_results_path, name))]
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = defaultdict(list)
    for video in tqdm(videos):
        if video in results:
            continue
        video_path = os.path.join(multivent_yt_path, f"{video}.mp4")

        finished_frames = []
        mvg_gt = multivent_g_ground_truth[video]["spatial"]

        for object in mvg_gt:
            frame_number = object["frame"]
            if frame_number not in finished_frames:
                try:
                    im = load_frame_as_pil(video_path, frame_number)
                    # im.thumbnail([1024,1024], Image.Resampling.LANCZOS)
                    # Run model to find bounding boxes
                    response = client.models.generate_content(
                        model='gemini-1.5-flash-latest',
                        contents=[prompt, im],
                        config = types.GenerateContentConfig(
                            system_instruction=bounding_box_system_instructions,
                            temperature=0.1,
                            # safety_settings=safety_settings,
                        )
                    )

                    objects = convert_coords_back(response.text, frame_number)
                    if video not in results:
                        results[video] = []
                    results[video] += objects
                    finished_frames.append(frame_number)
                except Exception as e:
                    print(e)
    
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: round(float(o), 3) if isinstance(o, np.floating) else o)






multivent_yt_path = "/data/multivent_yt_videos/"
multivent_g_results_path = "/data/multivent_processed/"
multivent_g_json_path = "/home/aiden/Documents/cs/multiVENT/data/multivent_g.json"

test_gemini(multivent_yt_path, multivent_g_results_path, multivent_g_json_path)