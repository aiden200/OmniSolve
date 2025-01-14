import os, cv2, torch
from PIL import Image
import io, base64
import google.generativeai as genai
import json
import math
from metadata_rag.test_data import *
from collections import defaultdict
import matplotlib.pyplot as plt




class Information_processor:
    def __init__(self, warnings, objectives, working_dir="temp"):
        self.status_report_messages = []
        self.world_beliefs = []
        self.warnings = warnings
        self.objectives = objectives
        self.summary = ""
        self.working_dir = working_dir
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def write_facts_and_beliefs_to_disk(self):
        world_belief_sheet = os.path.join(self.working_dir, "factual_sheet.txt")
        status_report_sheet = os.path.join(self.working_dir, "status_report_sheet.txt")

        with open(world_belief_sheet, 'w') as f:
            f.write("World Beliefs\n")
            for line in self.world_beliefs:
                f.write(f"{line}\n")
        
        with open(status_report_sheet, 'w') as f:
            f.write("Status Reports\n")
            for line in self.status_report_messages:
                f.write(f"{line}\n")

    def _extract_json_between_markers(self, text, start_marker="```json", end_marker="```"):
        try:
            # Find the start and end positions
            start_pos = text.index(start_marker) + len(start_marker)
            end_pos = text.index(end_marker, start_pos)
            
            # Extract and return the text between the markers
            return text[start_pos:end_pos].strip()
        except ValueError:
            print("Markers not found in the text.")
            return None
    
    def update_summary(self, new_information): # we could benchmark this to multivent
        
        prompt = f"You are tasked with maintaining a concise summary of an ongoing event. The current summary is: {self.summary}.\
            The new information provided is: {new_information}. Determine if this new information significantly changes or adds to the current summary. If so,\
                write a new, updated summary within one short paragraph. If the new information does not affect the summary, reply with '__NONE__'."
        response = self.model.generate_content(prompt)
        if "__NONE__" not in response.text:
            self.summary = response.text
    
    def status_report(self, new_information, timestamp):
        prompt = f"You are tasked with a one sentence status report. The current status reports are: {str(self.status_report_messages)}.\
            The new information provided is: {new_information}. Write a one sentence status report given the \
                timestamp: {timestamp}, in the format [TIMESTAMP]: [STATUS REPORT]"
        response = self.model.generate_content(prompt)
        self.status_report_messages.append(response.text)
    

    

    def calculate_entropy(self, frame):
        """Calculate entropy based on objects and relationships in a frame."""
        num_objects = len(frame["objects"])
        num_relationships = len(frame["relationships"])
        movements = len(frame["movements"])
        
        # Assign weights to different components (tune as needed)
        entropy = num_objects * 0.15 + num_relationships * 0.05 + movements * 0.8
        return entropy

    def select_key_frames(self, video_data, top_n=4):
        """Select the top N frames based on entropy."""
        
        frame_entropies = {
            frame_id-1: self.calculate_entropy(video_data["frames"][frame_id]) # -1 for the 0 index
            for frame_id in video_data["frames"]
            
        }
        
        # Sort frames by entropy and select top N
        sorted_frames = sorted(frame_entropies.items(), key=lambda x: x[1], reverse=True)
        return sorted([frame_id for frame_id, _ in sorted_frames[:top_n]])


    def draw_bounding_boxes(self, frame_data, image, output_path):

        # print(image.shape)
        height, width, _ = image.shape

        # Create a mapping of object names to their center points
        object_centers = {}

        for obj in frame_data["objects"]:
            bbox = obj["bbox"]
            name = obj["name"]

            ymin = int((bbox[0] / 1000) * height)
            xmin = int((bbox[1] / 1000) * width)
            ymax = int((bbox[2] / 1000) * height)
            xmax = int((bbox[3] / 1000) * width)

            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            object_centers[name] = (center_x, center_y)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for rel in frame_data["relationships"]:
            obj1 = rel["object_1"]
            obj2 = rel["object_2"]
            relationship = rel["relationship"]

            if obj1 in object_centers and obj2 in object_centers:
                center1 = object_centers[obj1]
                center2 = object_centers[obj2]

                # Draw a line between the centers of the related objects
                cv2.line(image, center1, center2, (255, 0, 0), 2)

                # Place the relationship label near the midpoint of the line
                mid_x = (center1[0] + center2[0]) // 2
                mid_y = (center1[1] + center2[1]) // 2
                cv2.putText(image, relationship, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save the output image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_rgb)
        # plt.axis('off')
        # plt.show()
        # input("Press Enter to continue...")


    def parse_video_data(self, video_data):
        parsed_data = {
            "frames": {},
            "video_summary": video_data.get("video_summary", "")
        }

        for frame_key, frame_content in video_data.items():
            if frame_key.startswith("frame_"):
                frame_number = int(frame_key.split("_")[1])
                parsed_data["frames"][frame_number] = {
                    "objects": [
                        {
                            "name": obj["name"],
                            "bbox": obj["bbox"]
                        }
                        for obj in frame_content.get("objects", [])
                    ],
                    "relationships": [
                        {
                            "object_1": rel["object_1"],
                            "relationship": rel["relationship"],
                            "object_2": rel["object_2"]
                        }
                        for rel in frame_content.get("relationships", [])
                    ],
                    "movements": frame_content.get("movements", []),
                    "summary": frame_content.get("summary", "")
                }

        return parsed_data
     

    def update_respective_informations(self, video_path, summary_of_clip):
        
        text, frames = self.dense_caption(video_path)
        text = TEST_DATA
        text = self._extract_json_between_markers(text)
        json_data = json.loads(text)
        parsed_data = self.parse_video_data(json_data)

        key_frames = self.select_key_frames(parsed_data)
        for frame_num in key_frames:
            frame = frames[frame_num]
            self.draw_bounding_boxes(parsed_data["frames"][frame_num+1], frame, "")

        return parsed_data
        
        # pass
        # a warning should update the status report
    

    def dense_caption(self, video_path):
        frames = self._split_frames_per_second(video_path)
        bounding_box_prompt = """
For each frame, perform the following:
1. Identify all objects in the image and return their bounding boxes in [ymin, xmin, ymax, xmax] format.
2. Describe the spatial relationships between the objects (e.g., "on top of," "next to").
3. Track the movement of objects compared to the previous frames and summarize their trajectory.
4. Generate a concise summary of the frame that includes the key objects, relationships, and actions.

After processing all frames:
5. Summarize the overall dynamics of the video, including the key objects, their interactions, and movement patterns.

Format the output in JSON as follows:
{
    "frame_n": {
        "objects": [{"name": "object_name", "bbox": [ymin, xmin, ymax, xmax]}],
        "relationships": [{"object_1": "name", "relationship": "relation", "object_2": "name"}],
        "movements": [{"object": "name", "trajectory": "description"}],
        "summary": "Frame-level summary here."
    },
    "video_summary": "Overall video summary here."
}
"""

        
        images_data = []
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG") 
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            images_data.append({'mime_type': 'image/jpeg', 'data': image_base64})
        
        images_data.append(bounding_box_prompt)
        
        # Generate dense captions using the model
        response = self.model.generate_content(images_data)
        return response.text, frames
    

    def _split_frames_per_second(self, video_path):
        raw_video = cv2.VideoCapture(video_path)
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        frame_count = 0
        frames = []

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            # Process only 1 frame per second
            if frame_count % frame_rate == 0:
                frames.append(raw_frame)

            frame_count += 1

        raw_video.release()
        return frames



if __name__ == "__main__":
    processor = Information_processor([], [])
    print(processor.update_respective_informations("/home/aiden/Documents/cs/OmniSolve/question_generation/trimmed_video.mp4", ""))

    