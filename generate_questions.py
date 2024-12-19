from dotenv import load_dotenv
import os
from question_generation.video_question_gen import VideoQuestionGenerator
from MMDuet.extract_timestamps import TimestampExtracter
from question_generation.prompts import *

load_dotenv()

class VideoParser:
    def __init__(self, prompt=DEFAULT_MM_PROMPT):
        self.processor = VideoQuestionGenerator()
        self.timestampExtracter = TimestampExtracter(prompt)
    

    def split_and_generate_video(self, video_url):
        if not os.path.exists(video_url):
            raise ValueError("Video url doesn't exist")
        folder = "generated_data"
        filename = os.path.splitext(os.path.basename(video_url))[0]
        new_folder = os.path.join(folder, filename)
        counter = 0
        start_time = 0
        

        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
                            
        for timestamp, response, informative_score, relevance_score, frame, additional_info in self.timestampExtracter.start_chat(video_url):
            if response:
                end_time = timestamp - 1 # once the llm switched to a new scene its too late
                print(f"Time {start_time} - {end_time}")
                vid_output_file_path = f"{new_folder}/{counter}_video.mp4"
                question_output_file_path = f"{new_folder}/{counter}_question.json"
                text_output_file_path = f"{new_folder}/{counter}_description.text"
                self.processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, question_output_file_path, text_output_file_path)
                start_time = end_time + 1
                counter += 1
        
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            print(f"Time {start_time} - {end_time}")
            vid_output_file_path = f"{new_folder}/{counter}_video.mp4"
            question_output_file_path = f"{new_folder}/{counter}_question.json"
            text_output_file_path = f"{new_folder}/{counter}_description.text"
            self.processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, question_output_file_path, text_output_file_path)
            start_time = end_time + 1
            counter += 1
    
    def generate_over_entire_video(self, video_url):
        pass


    
    def split_timestamps(self, mode=None, threshold_mode="sum", threshold=2.0):
        curr_score = 0

parser = VideoParser()
video_url = "/home/aiden/Documents/cs/OmniSolve/depth_extraction/train_derailment_scene1/trimmed_output.mp4"
video_url = "/home/aiden/Documents/cs/OmniSolve/question_generation/full_train_derailment.mp4"
parser.split_and_generate_video(video_url)

# start_time = 1  # Start timestamp in seconds
# end_time = 10   # End timestamp in seconds
# 
# response = processor.qa_over_part_video(video_url, start_time, end_time)
# print(response)