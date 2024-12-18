import os
import google.generativeai as genai
import time
from prompts import *
import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from dotenv import load_dotenv

load_dotenv()


class VideoQuestionGenerator:
    def __init__(self, model_name="gemini-1.5-pro"):
        self.model_name = model_name
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = genai.GenerativeModel(model_name=model_name)


    def _configure_video_file(self, video_path):
        print(f"Uploading file...")
        video_file = genai.upload_file(path=video_path)
        print(f"Completed upload: {video_file.uri}")
        return video_file


    def _check_process_video(self, video_file):
        # Check whether the file is ready to be used.
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)


    def qa_over_entire_video(self, video_path):
        video_file = self._configure_video_file(video_path)
        self._check_process_video(video_file)

        prompt = VIDEO_PROMPT
        response = self.model.generate_content([video_file, prompt],
                                    request_options={"timeout": 600})
        
        return response.text
    

    def qa_over_part_video(self, video_path, start_time, end_time):
        trimmed_video_path = "trimmed_video.mp4"
        ffmpeg_extract_subclip(video_path, start_time, end_time, outputfile=trimmed_video_path)
        video_file = self._configure_video_file(trimmed_video_path)
        self._check_process_video(video_file)

        prompt = VIDEO_PROMPT
        response = self.model.generate_content([video_file, prompt],
                                    request_options={"timeout": 600})
        
        return response.text


    def qa_over_timestamp_in_full_video(self, video_path, prompt_num=2):
        video_file = self._configure_video_file(video_path)
        self._check_process_video(video_file)

        if prompt_num == 1:
            prompt = VIDEO_PROMPT
        elif prompt_num == 2:
            prompt = VIDEO_PROMPT2
        response = self.model.generate_content([video_file, prompt],
                                    request_options={"timeout": 600})
        
        return response.text

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



    def parse_json_format(self, text, output_file="parsed_train_derailment_data.json"):
        text = self._extract_json_between_markers(text)
        parsed_data = json.loads(text)
        with open(output_file, 'w') as file:
            json.dump(parsed_data, file, indent=4)



processor = VideoQuestionGenerator()
start_time = 1  # Start timestamp in seconds
end_time = 10   # End timestamp in seconds
video_url = "/home/aiden/Documents/cs/OmniSolve/depth_extraction/train_derailment_scene1/trimmed_output.mp4"

response = processor.qa_over_part_video(video_url, start_time, end_time)
print(response)


    