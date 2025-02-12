import os, cv2, torch
from PIL import Image
import io, base64
import google.generativeai as genai
import json
import math
from metadata_rag.test_data import *
from collections import defaultdict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor





class Information_processor:
    def __init__(self, warnings, objectives, working_dir="temp"):
        self.status_report_messages = []
        self.long_term_memory = ""
        self.warnings = warnings
        self.objectives = objectives
        self.summary = ""
        self.working_dir = working_dir
        self.long_term_memory_file = os.path.join(working_dir, "long_term_memory.txt")
        self.status_report_file = os.path.join(working_dir, "status_reports.txt")
        self.summary_file = os.path.join(working_dir, "summary.txt")
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")


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
    
    def _update_summary(self, new_information): # we could benchmark this to multivent
        prompt = f"You are tasked with maintaining a concise summary of an ongoing event. The current summary is: {self.summary}.\
            The new information provided is: {new_information}. Given the objectives: {str(self.objectives)} and warnings: {str(self.warnings)}, Determine if this new information significantly changes or adds to the current summary. If so,\
                write a new, updated summary within one short paragraph. If the new information does not affect the summary, reply with '__NONE__'."
        response = self.model.generate_content(prompt)
        if "__NONE__" not in response.text:
            self.summary = response.text
            with open(self.summary_file, "w") as f:
                f.write(self.summary)
    
    def _status_report(self, new_information, timestamp, objectives_updates, warnings_updates):
        prompt = f"You are tasked with a one sentence status report. The current status reports are: {str(self.status_report_messages)}.\
            The new information provided is: {new_information}. Given the objectives: {str(self.objectives)} and warnings: {str(self.warnings)}, Write a one sentence status report given the \
                timestamp: {timestamp}, in the format [TIMESTAMP]: [STATUS REPORT]. only return the newest status report. Do not used the format OBJECTIVE RELATED: or WARNING RELATED: updates, those are manual updates from humans."
        

        with open(self.status_report_file, "a") as f:
            if objectives_updates:
                self.status_report_messages.append(f"OBJECTIVE RELATED: {objectives_updates}")
                f.write(f"OBJECTIVE RELATED: {objectives_updates}" + "\n\n")
            
            if warnings_updates:
                for warnings in warnings_updates:
                    self.status_report_messages.append(f"WARNING RELATED: {warnings}")
                    f.write(f"WARNING RELATED: {warnings}" + "\n\n")
            
            response = self.model.generate_content(prompt)
            self.status_report_messages.append(response.text)
            f.write(response.text + "\n")

    

    def _update_long_term_memory(self, new_information, new_object_information): # this might not actually add any value. we'll see
        prompt = (
            f"You are tasked with maintaining a concise set of world facts. The current long-term memory is: {str(self.long_term_memory)}. \
            The new information provided is: {new_information}. The new object relationships provided are: {new_object_information}. \
            Given the objectives: {str(self.objectives)} and warnings: {str(self.warnings)}, refine the long-term memory. \
            Ensure it aligns with the objectives, detects the warnings, and remains concise. Write the updated long-term memory."
        )
        response = self.model.generate_content(prompt)
        self.long_term_memory = response.text
        with open(self.long_term_memory_file, "w") as f:
            f.write(self.long_term_memory)

    
    def execute_parallel_updates(self, new_information, timestamp, new_object_information, objectives_updates=None, warnings_updates=None):
        """
        Executes update_summary, status_report, and update_long_term_memory in parallel.
        
        Parameters:
            new_information (str): New information to process.
            timestamp (str): The timestamp for the status report.
            new_object_information (str): Additional object-related information for long-term memory.
        
        Returns:
            None
        """
        with ThreadPoolExecutor() as executor:
            
            future_update_summary = executor.submit(self._update_summary, new_information)
            future_status_report = executor.submit(self._status_report, new_information, timestamp, objectives_updates, warnings_updates)
            future_update_long_term_memory = executor.submit(self._update_long_term_memory, new_information, new_object_information)
            
            
            for future in [future_update_summary, future_status_report, future_update_long_term_memory]:
                future.result() 


    def process_q_and_a(self, question, text_context, object_context):
        PROMPT_TEMPLATE = f"""
Answer the question based only on the following context:
{text_context}


 - -
If you cannot answer the question based on the context, reply 'I don't know'.
Answer the question based on the above context: {question}
"""
        answer = self.model.generate_content(PROMPT_TEMPLATE)

        return answer.text
    

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
            frame_id: self.calculate_entropy(video_data["frames"][frame_id]) # -1 for the 0 index
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

            # ymin = int((bbox[0] / 1000) * height)
            # xmin = int((bbox[1] / 1000) * width)
            # ymax = int((bbox[2] / 1000) * height)
            # xmax = int((bbox[3] / 1000) * width)

            # center_x = (xmin + xmax) // 2
            # center_y = (ymin + ymax) // 2
            # object_centers[name] = (center_x, center_y)

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ymin, xmin, ymax, xmax = bbox
            x1 = int(xmin / 1000 * width)
            y1 = int(ymin / 1000 * height)
            x2 = int(xmax / 1000 * width)
            y2 = int(ymax / 1000 * height)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            object_centers[name] = (center_x, center_y)
            
            cv2.putText(image, name, (x1+2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


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


    # def parse_video_data(self, video_data):
    #     parsed_data = {
    #         "frames": {},
    #         "video_summary": video_data.get("video_summary", "")
    #     }

    #     for frame_key, frame_content in video_data.items():
    #         if frame_key.startswith("frame_"):
    #             frame_number = int(frame_key.split("_")[1])
    #             parsed_data["frames"][frame_number] = {
    #                 "objects": [
    #                     {
    #                         "name": obj["name"],
    #                         "bbox": obj["bbox"]
    #                     }
    #                     for obj in frame_content.get("objects", [])
    #                 ],
    #                 "relationships": [
    #                     {
    #                         "object_1": rel["object_1"],
    #                         "relationship": rel["relationship"],
    #                         "object_2": rel["object_2"]
    #                     }
    #                     for rel in frame_content.get("relationships", [])
    #                 ],
    #                 "movements": frame_content.get("movements", []),
    #                 "summary": frame_content.get("summary", "")
    #             }

    #     return parsed_data
     

    def parse_video_data(self, video_data):
        parsed_data = {
            "frames": {},
            "video_summary": video_data.get("video_summary", "")
        }

        objects = []

        for frame_key, frame_content in video_data.items():
            if frame_key.startswith("frame_"):
                frame_number = int(frame_key.split("_")[1])

                # Track how many times each name appears in this frame
                name_count = {}

                renamed_objects = []
                for obj in frame_content.get("objects", []):
                    original_name = obj["name"]
                    bbox = obj["bbox"]
                    
                    count = name_count.get(original_name, 0)
                    # If you want the first occurrence to remain as "human"
                    # and subsequent ones to be "human_1", "human_2", etc.:
                    if count == 0:
                        new_name = original_name  # or f"{original_name}_0" if you prefer
                    else:
                        new_name = f"{original_name}_{count}"

                    # Update the counter
                    name_count[original_name] = count + 1

                    renamed_objects.append({"name": new_name, "bbox": bbox})
                    if new_name not in objects:
                        objects.append(new_name)

                parsed_data["frames"][frame_number] = {
                    "objects": renamed_objects,
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

        return parsed_data, objects


    def update_respective_information(self, video_path):
        
        text, frames = self.dense_caption(video_path)
        # text = TEST_DATA
        text = self._extract_json_between_markers(text)
        json_data = json.loads(text)
        parsed_data, objects = self.parse_video_data(json_data)

        key_frames = self.select_key_frames(parsed_data)
        i = 0
        for frame_num in key_frames:
            frame = frames[frame_num] # this line is wrong
            output_name = f"{video_path[:-4]}_{i}.jpg"
            self.draw_bounding_boxes(parsed_data["frames"][frame_num], frame, output_name)
            i += 1

        return parsed_data, objects
        
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

Format the output in JSON as follows, where "frame_n" starts with "frame_0":
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

    