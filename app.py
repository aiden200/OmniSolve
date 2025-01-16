
import threading
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os, shutil
from queue import Queue
import cv2

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
    def __init__(self, rag_system: RAGSystem, information_processor: Information_processor, video_question_generator: VideoQuestionGenerator):
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


                current_video_description = self.video_question_generator.qa_over_part_video(
                    video_url, start_time, end_time,
                    vid_output_file_path, text_output_file_path,
                    prev_context=self.prev_context
                )

                self.prev_context.append(current_video_description) 

                # Get objects from dense captioning & Extract Important clips TODO: Paralize 
                new_object_information, new_objects = self.information_processor.update_respective_information(vid_output_file_path)
                
                # match any warning object alerts
                warnings_messages = self.rag.detect_warnings(new_objects, 0.6)
                
                # Populate the regular vector DB & match any alerts from Objectives or warnings
                self.rag.add_to_rag(current_video_description, vid_id)
                
                # TODO: Populate the object KG
                # self.rag.add_to_kg(new_object_information, vid_id)

                
                # TODO: Done - Obtain the depth 
                # depth_output_path = f"{vid_output_file_path[:-4]}_d.mp4"
                # self.dpt.extract_video_depth(vid_output_file_path, depth_output_path)

                # TODO: Populate the 3D SG


                # update the texts
                self.information_processor.execute_parallel_updates(
                    current_video_description, 
                    f'{start_time} - {end_time}', 
                    new_object_information,
                    objectives_updates,
                    warnings_messages
                    )



            except Exception as e:
                print(f"Vector population error: {e}")
                log.info(f"ERROR: Vector population error for {self.task_num} with error: {e}")
                self.task_num += 1
                self.task_queue.task_done()
            finally:
                log.info(f"Vector population for task {self.task_num} completed")
                self.task_num += 1
                self.task_queue.task_done()

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
    def __init__(self, rag_system: RAGSystem, information_processor: Information_processor, timestamp_extractor: TimestampExtracter, video_question_generator: VideoQuestionGenerator, working_dir):
        self.rag = rag_system
        self.timestampExtracter = timestamp_extractor
        self.working_dir = working_dir
        self.vector_population_worker = VectorPopulatorWorker(rag_system=rag_system, information_processor=information_processor, video_question_generator=video_question_generator)
        

    def real_time_video_process(self, video_url, output_dir):
        if not os.path.exists(video_url):
            raise ValueError("Video url doesn't exist")
        
        log.info("Starting real time video process")
        
        counter = 0
        start_time = 0
        end_time = 0


        for timestamp, response, informative_score, relevance_score, frame, additional_info in self.timestampExtracter.start_chat(video_url):
            # print("frame is in")
            # if frame is not None:
            #     cv2.imshow("Debug Video Playback", frame)
                # A waitKey of ~30 ms matches ~30fps. Adjust as needed.
                # If user presses 'q', we'll exit the loop.
                # if cv2.waitKey(30) & 0xFF == ord('q'):
                #     print("User requested exit from debug video window.")
                #     break
            
            if response:
                end_time = timestamp - 1  # once the llm switched to a new scene it's too late
                vid_output_file_path = os.path.join(output_dir, f"{counter}.mp4")
                text_output_file_path = os.path.join(output_dir, f"{counter}.txt")

                self.vector_population_worker.add_task(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, counter)

                start_time = end_time + 1
                counter += 1
        
        # Process any leftover from [end_time+1, final timestamp]
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            vid_output_file_path = os.path.join(output_dir, f"{counter}.mp4")
            text_output_file_path = os.path.join(output_dir, f"{counter}.txt")

            self.vector_population_worker.add_task(video_url, start_time, end_time, vid_output_file_path, text_output_file_path, counter)

            start_time = end_time + 1
            counter += 1
        
        # cv2.destroyAllWindows()


class SharedResources:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedResources, cls).__new__(cls)
            cls._instance.initialize_resources()
        return cls._instance

    def initialize_resources(self):
        log.info("Initializing shared resources")
        with open(OBJECTIVE_FILE, 'r') as f:
            objectives = f.read()

        warnings = []
        with open(WARNINGS_FILE, "r") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    warnings.append(stripped_line)
        

        with open(STORE_OBJECTIVE_FILE, 'w') as f:
            f.write(objectives)

        
        with open(STORE_WARNINGS_FILE, "w") as f:
            for warn in warnings:
                f.write(warn + "\n")

        TIMESTAMP_EXTRACTOR = TimestampExtracter(DEFAULT_MM_PROMPT)
        log.info("Timestamp Extractor loaded")
        INFORMATION_PROCESSOR = Information_processor(warnings=warnings, objectives=objectives, working_dir=WORKING_DIR)
        log.info("Information Processor loaded")
        VIDEO_QUESTION_GENERATOR = VideoQuestionGenerator()
        log.info("Video Question Generator loaded")
        RAG_SYSTEM = RAGSystem(db_dir=DB_DIR, warnings=warnings)
        log.info("RAG Loaded")
        RTP = RealTimeVideoProcess(rag_system=RAG_SYSTEM,
                                    information_processor=INFORMATION_PROCESSOR, 
                                    timestamp_extractor=TIMESTAMP_EXTRACTOR, 
                                    video_question_generator=VIDEO_QUESTION_GENERATOR, 
                                    working_dir=WORKING_DIR)
        log.info("RTP Loaded")

        self.timestamp_extractor = TIMESTAMP_EXTRACTOR
        self.information_processor = INFORMATION_PROCESSOR
        self.video_question_generator = VIDEO_QUESTION_GENERATOR
        self.rag_system = RAG_SYSTEM
        self.rtp = RTP


# ----------------------------------
# FLASK APP
# ----------------------------------
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Video QA Flask App is running!"



@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Endpoint to receive a video file, store it locally,
    and start the background process for chunked QA + DB insertion.
    """
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

    resources = SharedResources()  # Get the shared resources

    RTP = resources.rtp



    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save file locally in 'uploads' folder (make sure folder exists)
        upload_folder = WORKING_DIR
        
        video_path = os.path.join(upload_folder, filename)
        file.save(video_path)

        # Create an output folder for chunked videos/text
        output_dir = os.path.join(upload_folder, "generated_video_content")
        os.makedirs(output_dir, exist_ok=True)
        RTP.real_time_video_process(video_path, output_dir)

        # # Run the real_time_video_process in a new Thread
        # def background_task():
        #     try:
        #     except Exception as e:
        #         print(f"Background task error: {e}")

        # thread = threading.Thread(target=background_task, daemon=True)
        # thread.start()

        return jsonify({"message": f"Video {filename} uploaded. Processing started in background."}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/query', methods=['POST'])
def query_vector_db():
    """
    Endpoint to let a user ask questions. We do a naive search
    in our vector DB and return the best match (or top K).
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    question = data['question']

    resources = SharedResources()  # Get the shared resources
    RAG_SYSTEM = resources.rag_system
    

    # Get data from regular summarization RAG
    relevant_documents = RAG_SYSTEM.query_vector_store(question)


    image_paths = []
    video_path = None
    if relevant_documents == "__NONE__":
        return jsonify({"answer": "No data in vector DB yet.", "video_path": video_path, "image_paths": image_paths})
    

    if relevant_documents == "__NONE__":
        return jsonify({"answer": "RAG not populated yet!"})
    
    # Get the specific video clips associated and the respective entropy frames (just the paths)
    most_relevant_vid_id = relevant_documents[0]['metadata']['full_video_id']
    video_path = os.path.join(WORKING_DIR, "generated_video_content", f"{most_relevant_vid_id}.mp4")
    for i in range(4):
        name = os.path.join(WORKING_DIR, "generated_video_content", f"{most_relevant_vid_id}_{i}.mp4")
        if os.path.exists(name):
            image_paths.append(name)
        else:
            image_paths.append("")

    # TODO: Get the KG Rag and the specific entity relationships (maybe have a UI for this)
    relevant_kg_doucments = RAG_SYSTEM.query_kg(question)

    # TODO: Use this to generate a comprehensive answer and return this to the user, along with the evidence.
    answer = ""
    # TODO: I need a simple UI on the other side of the process to be able to handle this.



    return jsonify({
        "answer": answer,
        "video_path": video_path,
        "image_paths": image_paths
        })


# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == '__main__':
    app.run(debug=True)
