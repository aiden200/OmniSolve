from dotenv import load_dotenv
import os, shutil
from question_generation.video_question_gen import VideoQuestionGenerator
from MMDuet.extract_timestamps import TimestampExtracter
from metadata_rag.information_processor import Information_processor 
from question_generation.prompts import *


load_dotenv()

class VideoParser:
    def __init__(self, prompt=DEFAULT_MM_PROMPT, load_timestamp_processer=True, warnings=[], objectives=[], working_dir="temp"):
        # self.video_processor = VideoQuestionGenerator()
        self.information_processor = Information_processor(warnings, objectives, working_dir)
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)
        # if load_timestamp_processer:
        #     self.timestampExtracter = TimestampExtracter(prompt)

    
    def summarize_video_given_text(self, output_file, texts):
        self.video_processor.summarize_text(output_file, texts)

    def split_and_generate_video(self, video_url, qa=False):
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

        prev_context = []

        for timestamp, response, informative_score, relevance_score, frame, additional_info in self.timestampExtracter.start_chat(video_url):
            if response:
                end_time = timestamp - 1 # once the llm switched to a new scene its too late
                print(f"Time {start_time} - {end_time}")
                vid_output_file_path = f"{new_folder}/{counter}_video.mp4"
                question_output_file_path = f"{new_folder}/{counter}_question.json"
                text_output_file_path = f"{new_folder}/{counter}_description.text"
                curr_context = self.video_processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, question_output_file_path, text_output_file_path, qa=qa, prev_context=prev_context)
                prev_context.append(curr_context)
                start_time = end_time + 1
                counter += 1
        
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            print(f"Time {start_time} - {end_time}")
            vid_output_file_path = f"{new_folder}/{counter}_video.mp4"
            question_output_file_path = f"{new_folder}/{counter}_question.json"
            text_output_file_path = f"{new_folder}/{counter}_description.text"
            self.video_processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, question_output_file_path, text_output_file_path,qa=qa)
            start_time = end_time + 1
            counter += 1
    

    def real_time_video_process(self, video_url, new_url):
        if not os.path.exists(video_url):
            raise ValueError("Video url doesn't exist")

        counter = 0
        start_time = 0


        prev_context = []

        for timestamp, response, informative_score, relevance_score, frame, additional_info in self.timestampExtracter.start_chat(video_url):
            if response:
                end_time = timestamp - 1 # once the llm switched to a new scene its too late
                vid_output_file_path = os.path.join(new_url, f"{counter}.mp4")
                text_output_file_path = os.path.join(new_url, f"{counter}.txt")
                curr_context = self.video_processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, prev_context=prev_context)
                prev_context.append(curr_context)
                start_time = end_time + 1
                counter += 1
        
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            vid_output_file_path = os.path.join(new_url, f"{counter}.mp4")
            text_output_file_path = os.path.join(new_url, f"{counter}.txt")
            self.video_processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, prev_context=prev_context)
            start_time = end_time + 1
            counter += 1


    def split_and_store_vids(self, video_url, new_url):
        if not os.path.exists(video_url):
            raise ValueError("Video url doesn't exist")

        counter = 0
        start_time = 0


        prev_context = []

        for timestamp, response, informative_score, relevance_score, frame, additional_info in self.timestampExtracter.start_chat(video_url):
            if response:
                end_time = timestamp - 1 # once the llm switched to a new scene its too late
                vid_output_file_path = os.path.join(new_url, f"{counter}.mp4")
                text_output_file_path = os.path.join(new_url, f"{counter}.txt")
                curr_context = self.video_processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, prev_context=prev_context)
                prev_context.append(curr_context)
                start_time = end_time + 1
                counter += 1
        
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            vid_output_file_path = os.path.join(new_url, f"{counter}.mp4")
            text_output_file_path = os.path.join(new_url, f"{counter}.txt")
            self.video_processor.qa_over_part_video(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, prev_context=prev_context)
            start_time = end_time + 1
            counter += 1
    

    def dense_caption_videos(self, video_path):
        dense_captions = self.information_processor.update_respective_informations(video_path, "")
        print(dense_captions)

    def generate_over_entire_video(self, video_url):
        pass


    
    def split_timestamps(self, mode=None, threshold_mode="sum", threshold=2.0):
        curr_score = 0

# parser = VideoParser()
# video_url = "/home/aiden/Documents/cs/OmniSolve/depth_extraction/train_derailment_scene1/trimmed_output.mp4"
# video_url = "/home/aiden/Documents/cs/OmniSolve/question_generation/full_train_derailment.mp4"
# parser.split_and_generate_video(video_url)
v = VideoParser(load_timestamp_processer=False)
v.dense_caption_videos("/home/aiden/Documents/cs/OmniSolve/question_generation/trimmed_video.mp4")

# start_time = 1  # Start timestamp in seconds
# end_time = 10   # End timestamp in seconds
# 
# response = processor.qa_over_part_video(video_url, start_time, end_time)
# print(response)