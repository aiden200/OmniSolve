
import threading
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os, shutil
from queue import Queue
import cv2
import json

from config import *
from question_generation.video_question_gen import VideoQuestionGenerator
from MMDuet.extract_timestamps import TimestampExtracter
from depth_extraction.extract_depth import DepthCalculator
from metadata_rag.information_processor import Information_processor 
from metadata_rag.vqa_rag import RAGSystem
from question_generation.prompts import *
import logging


log_file_path = "app.log"
if os.path.exists(log_file_path):
    os.remove(log_file_path)

logging.basicConfig(
    level=logging.INFO,  # or logging.INFO for less verbosity
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file_path)]
)


log = logging.getLogger(__name__)


load_dotenv()



class VectorPopulatorWorker:
    def __init__(self, rag_system: RAGSystem, information_processor: Information_processor, video_question_generator: VideoQuestionGenerator, description=None):
        # self.dpt = DepthCalculator()
        self.task_queue = Queue()
        self.stop_event = threading.Event()
        self.rag = rag_system
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.task_num = 0
        self.thread.start()
        self.information_processor = information_processor
        self.video_question_generator = video_question_generator
        self.prev_context = []
        if description:
            self.description = description

    def add_task(self, video_url, start_time, end_time, vid_output_file_path, text_output_file_path, vid_id):
        """Add a text chunk to the queue for embedding."""
        self.task_queue.put([video_url, start_time, end_time, vid_output_file_path, text_output_file_path, vid_id])

    def _run(self):
        """Continuously process tasks from the queue."""
        while not self.stop_event.is_set():
            try:
                [video_url, start_time, end_time, vid_output_file_path, text_output_file_path, vid_id] = self.task_queue.get(timeout=1)
            except:
                # If queue is empty for a while, loop back to check stop_event
                continue

            # Now embed + insert into vector DB
            try:
                # This might be expensive and take a while
                log.info(f"Vector population for task {self.task_num} for timestamp {start_time} started.")
                objectives_updates, warnings_updates = None, None

                # We want the last thing the model sees the description
                if self.description:
                    self.prev_context.append(self.description)
                    
                current_video_description = self.video_question_generator.qa_over_part_video(
                    video_url, start_time, end_time,
                    vid_output_file_path, text_output_file_path,
                    prev_context=self.prev_context
                )

                if self.description:
                    self.prev_context[-1] = current_video_description
                else:
                    self.prev_context.append(current_video_description) 

                # Get objects from dense captioning & Extract Important clips TODO: Paralize 
                # new_object_information, new_objects = self.information_processor.update_respective_information(vid_output_file_path)
                
                # # match any warning object alerts
                # warnings_messages = self.rag.detect_warnings(new_objects, 0.6)
                
                
                # # Populate the regular vector DB & match any alerts from Objectives or warnings
                self.rag.add_to_rag(current_video_description, vid_id)
                
                # # TODO: Populate the object KG
                # # self.rag.add_to_kg(new_object_information, vid_id)

                
                # # TODO: Done - Obtain the depth 
                # # depth_output_path = f"{vid_output_file_path[:-4]}_d.mp4"
                # # self.dpt.extract_video_depth(vid_output_file_path, depth_output_path)

                # # TODO: Populate the 3D SG


                # # update the texts
                # self.information_processor.execute_parallel_updates(
                #     current_video_description, 
                #     f'{start_time} - {end_time}', 
                #     new_object_information,
                #     objectives_updates,
                #     warnings_messages
                #     )



            except Exception as e:
                print(f"Vector population error: {e}")
                log.info(f"ERROR: Vector population error for task {self.task_num} with error: {e}")
                self.task_num += 1
                if self.task_queue.not_empty:
                    self.task_queue.task_done()
            finally:
                log.info(f"Vector population for task {self.task_num} completed")
                self.task_num += 1
                if self.task_queue.not_empty:
                    self.task_queue.task_done()
                # self.task_queue.task_done()

    def stop(self):
        """Signal the worker to stop, then wait for thread to finish."""
        self.stop_event.set()
        self.thread.join()


# ----------------------------------
# REAL-TIME PROCESSING CLASS
# ----------------------------------
class RealTimeVideoProcess:
    """
    Your main orchestrator that processes the video chunk by chunk,
    storing text data in the vector DB as soon as it appears.
    """
    def __init__(self, rag_system: RAGSystem, information_processor: Information_processor, timestamp_extractor: TimestampExtracter, video_question_generator: VideoQuestionGenerator, working_dir, description):
        self.rag = rag_system
        self.timestampExtracter = timestamp_extractor
        self.working_dir = working_dir
        self.vector_population_worker = VectorPopulatorWorker(rag_system=rag_system, information_processor=information_processor, video_question_generator=video_question_generator, description=description)
    


    def real_time_video_process(self, video_url, output_dir, previous_context=None):
        if not os.path.exists(video_url):
            raise ValueError("Video url doesn't exist")
        
        log.info("Starting real time video process")
        
        counter = 0
        start_time = 0
        end_time = 0

        frame_output_filepath = os.path.join(output_dir, f"frame.txt")
        with open(frame_output_filepath, "w") as f:
            f.write("")

        for timestamp, response, informative_score, relevance_score, frame, frame_idx, frame_fps in self.timestampExtracter.start_chat(video_url):
            # print("frame is in")
            # if frame is not None:
            #     cv2.imshow("Debug Video Playback", frame)
                # A waitKey of ~30 ms matches ~30fps. Adjust as needed.
                # If user presses 'q', we'll exit the loop.
                # if cv2.waitKey(30) & 0xFF == ord('q'):
                #     print("User requested exit from debug video window.")
                #     break
            with open(frame_output_filepath, "a") as f:
                f.write(f"{counter},{timestamp},{frame_idx},{frame_fps}\n")
            
            if response:
                end_time = timestamp  # once the llm switched to a new scene it's too late
                vid_output_file_path = os.path.join(output_dir, f"{counter}.mp4")
                text_output_file_path = os.path.join(output_dir, f"{counter}.txt")

                self.vector_population_worker.add_task(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, counter)

                start_time = end_time
                counter += 1
        
        # Process any leftover from [end_time+1, final timestamp]
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            vid_output_file_path = os.path.join(output_dir, f"{counter}.mp4")
            text_output_file_path = os.path.join(output_dir, f"{counter}.txt")

            self.vector_population_worker.add_task(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, counter)

            start_time = end_time
            counter += 1
        
        log.info("waiting for the completion of vector population worker")
        self.wait_for_completion()
        # cv2.destroyAllWindows()
    
    def wait_for_completion(self):
        """Wait for the vector population worker to complete all queued tasks."""
        self.vector_population_worker.task_queue.join()
        log.info("All vector population tasks completed.")


ALLOWED_EXTENSIONS = {'mp4'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------------------------
# MAIN
# ----------------------------------

def write_to_check_file(checkfile, status, new_element):
    with open(checkfile, "a") as file:
        file.write(f"{new_element},{str(status)}\n")


def run_pipeline(video_dir, output_dir, json_file=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    check_file = os.path.join(output_dir, "check.txt")
    
    lines = []
    
    if os.path.exists(check_file):
        with open(check_file, "r") as file:
            lines = file.readlines()
        lines = [line.split(",")[0] for line in lines]
    

    # We only want to load this once
    te = TimestampExtracter(DEFAULT_MM_PROMPT)


    if json_file:
        with open(json_file, 'r') as f:
            data = json.load(f)
        for video in data:
            if video in lines:
                print(f"Video {video} already completed operation")
                continue
            try:
                video_path = os.path.join(video_directory, f"{video}.mp4")
                if not os.path.exists(video_path):
                    write_to_check_file(check_file, 1, video)
                    print(f"Video {video_path} does not exist")
                    continue
                
                description = data[video]["description"]
                language = data[video]["metadata"]["language"]
                category = data[video]["event_type"]
                
                if category != "emergency" or language != "english":
                    write_to_check_file(check_file, 2, video)
                    continue

                
                video_dir = os.path.join(output_dir, video)
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                
                db_dir = os.path.join(video_dir, "DB")

                # grab description
                

                ip = Information_processor(warnings=[], objectives="", working_dir=video_dir)
                log.info("Information Processor loaded")
                vqg = VideoQuestionGenerator()
                log.info("Video Question Generator loaded")
                rs = RAGSystem(db_dir=db_dir)
                log.info("RAG Loaded")
                RTP = RealTimeVideoProcess(rag_system=rs,
                                            information_processor=ip, 
                                            timestamp_extractor=te, 
                                            video_question_generator=vqg, 
                                            working_dir=video_dir,
                                            description=description)
                
                
                RTP.real_time_video_process(video_path, video_dir, description)
                
                
                # 0, Success, 1 Video Doesn't exist, 2 wrong category
                write_to_check_file(check_file, 0, video)
                
            except Exception as e:
                print(f"Failed video {video} with exception: {e}")
    else:
        # Just process videos in dir
        descriptions = {
            "DOD_110792773-1920x1080-9000k": "Some damage from the Palisades Fire in 2025. Soldiers are working hand in hand with the Los Angeles Fire Department and law enforcement agencies to ensure areas are secure and only proper personnel can enter certain areas for safety reasons.",
            "DOD_110804563-1920x1080-9000k": "BROLL Footage from Maui and Los Angeles. The U.S. Army Corp of Engineers uses the wet method to minimize the risk of ash and dust particles from entering the air during the debris removal process at residences in Los Angeles in the aftermath of the recent wildfires. This wet method process is done to protect the health and environment in the community.",
            "DOD_110770956-1920x1080-9000k": "California Army National Guard pilots fly over Palisades on a UH-60 Black Hawk observing damage to communities caused by the wildfires at Palisades, Calif., Jan. 15, 2025. The California Army National Guard was activated to help support first responders and emergency services fighting the fires in Los Angeles County. (U.S. Army National Guard video by Spc. William Franco Espinosa)"
        }
        videos = os.listdir(video_dir)
        for video in videos:
            video_path = os.path.join(video_dir, video)
            video_name = video[:-4]
            if video_name in lines:
                print(f"Video {video_name} already completed operation")
                continue
            try:
                
                description = descriptions[video_name]
                
                output_video_dir = os.path.join(output_dir, video_name)
                if not os.path.exists(output_video_dir):
                    os.mkdir(output_video_dir)
                
                db_dir = os.path.join(output_video_dir, "DB")

                # grab description
                

                ip = Information_processor(warnings=[], objectives="", working_dir=output_video_dir)
                log.info("Information Processor loaded")
                vqg = VideoQuestionGenerator()
                log.info("Video Question Generator loaded")
                rs = RAGSystem(db_dir=db_dir)
                log.info("RAG Loaded")
                RTP = RealTimeVideoProcess(rag_system=rs,
                                            information_processor=ip, 
                                            timestamp_extractor=te, 
                                            video_question_generator=vqg, 
                                            working_dir=output_video_dir,
                                            description=description)
                
                
                RTP.real_time_video_process(video_path, output_video_dir, description)
                
                
                # 0, Success, 1 Video Doesn't exist, 2 wrong category
                write_to_check_file(check_file, 0, video_name)
                
            except Exception as e:
                print(f"Failed video {video} with exception: {e}")



if __name__ == '__main__':
    video_directory = "/data/multivent_yt_videos/"
    json_file = "/home/aiden/Documents/cs/multiVENT/data/multivent_g.json"
    # output_dir = "/data/multivent_processed"
    output_dir = "/data/multivent_processed_without_delay"

    video_directory = "/data/disaster_videos/"
    output_dir = "/data/disaster_videos_processed/"
    run_pipeline(video_directory, output_dir)

    
    


    
