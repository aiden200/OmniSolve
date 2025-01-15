
import threading
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os, shutil
from queue import Queue
import cv2

from question_generation.video_question_gen import VideoQuestionGenerator
from MMDuet.extract_timestamps import TimestampExtracter
from metadata_rag.information_processor import Information_processor 
from question_generation.prompts import *
import logging




load_dotenv()



class VectorPopulatorWorker:
    def __init__(self, vector_db, information_processor: Information_processor, ):
        self.vector_db = vector_db
        self.task_queue = Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.task_num = 0
        self.thread.start()
        self.information_processor = information_processor

    def add_task(self, text_to_embed, timestamp):
        """Add a text chunk to the queue for embedding."""
        self.task_queue.put([text_to_embed, timestamp])

    def _run(self):
        """Continuously process tasks from the queue."""
        while not self.stop_event.is_set():
            try:
                [text_to_embed, timestamp] = self.task_queue.get(timeout=1)
            except:
                # If queue is empty for a while, loop back to check stop_event
                continue

            # Now embed + insert into vector DB
            try:
                # This might be expensive and take a while
                log.info(f"Vector population for task {self.task_num} for timestamp {timestamp} started.")
                ## TODO: Implement logic here
                self.vector_db.insert_text(text_to_embed) 
            except Exception as e:
                print(f"Vector population error: {e}")
                log.info(f"ERROR: Vector population error for {self.task_num} with error: {e}")
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
    def __init__(self, information_processor: Information_processor, timestamp_extractor: TimestampExtracter, video_question_generator: VideoQuestionGenerator, working_dir):

        self.information_processor = information_processor
        self.timestampExtracter = timestamp_extractor
        self.video_question_generator = video_question_generator
        self.working_dir = working_dir

    def real_time_video_process(self, video_url, output_dir):
        if not os.path.exists(video_url):
            raise ValueError("Video url doesn't exist")
        
        counter = 0
        start_time = 0
        end_time = 0

        prev_context = []

        for timestamp, response, informative_score, relevance_score, frame, additional_info in self.timestampExtracter.start_chat(video_url):
            if frame is not None:
                cv2.imshow("Debug Video Playback", frame)
                # A waitKey of ~30 ms matches ~30fps. Adjust as needed.
                # If user presses 'q', we'll exit the loop.
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    print("User requested exit from debug video window.")
                    break
            
            if response:
                end_time = timestamp - 1  # once the llm switched to a new scene it's too late
                vid_output_file_path = os.path.join(output_dir, f"{counter}.mp4")
                text_output_file_path = os.path.join(output_dir, f"{counter}.txt")

                # Actually process the chunk
                curr_context = self.video_question_generator.qa_over_part_video(
                    video_url, start_time, end_time,
                    vid_output_file_path, text_output_file_path,
                    prev_context=prev_context
                )
                prev_context.append(curr_context)

                # -----------------------------------------------
                # PARALLEL: populate vector DB with 'curr_context'
                # -----------------------------------------------
                # In a real scenario, do embedding => store in DB
                self.vector_db.insert_text(curr_context)

                start_time = end_time + 1
                counter += 1
        
        # Process any leftover from [end_time+1, final timestamp]
        if end_time != timestamp:
            start_time = end_time
            end_time = timestamp
            vid_output_file_path = os.path.join(output_dir, f"{counter}.mp4")
            text_output_file_path = os.path.join(output_dir, f"{counter}.txt")

            curr_context = self.video_processor.qa_over_part_video(
                video_url, start_time, end_time,
                vid_output_file_path, text_output_file_path,
                prev_context=prev_context
            )
            self.vector_db.insert_text(curr_context)
            prev_context.append(curr_context)

            start_time = end_time + 1
            counter += 1
        
        cv2.destroyAllWindows()


# ----------------------------------
# FLASK APP
# ----------------------------------
app = Flask(__name__)

temp_objective_path = "objectives.txt"
temp_warning_path = "warning.txt"

WORKING_DIR = "video_qa"
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
os.mkdir(WORKING_DIR)

DB_DIR = os.path.join(WORKING_DIR, "DB")
log_file_path = os.path.join(WORKING_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,  # or logging.INFO for less verbosity
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file_path)]
)

log = logging.getLogger(__name__)

OBJECTIVE_FILE = temp_objective_path
WARNINGS_FILE = temp_warning_path

with open(OBJECTIVE_FILE, 'w') as f:
    objectives = f.read()
log.info("Objectives loaded")

warnings = []
with open(WARNINGS_FILE, "w") as f:
    for line in f:
        stripped_line = line.strip()
        if stripped_line:
            warnings.append(stripped_line)

TIMESTAMP_EXTRACTOR = TimestampExtracter(DEFAULT_MM_PROMPT)
log.info("Timestamp Extractor loaded")
INFORMATION_PROCESSOR = Information_processor(warnings=warnings, objectives=objectives, working_dir=WORKING_DIR)
log.info("Information Processor loaded")
VIDEO_QUESTION_GENERATOR = VideoQuestionGenerator()
log.info("Video Question Generator loaded")




log.info("Warnings loaded")


with open(os.path.join(WORKING_DIR, "objectives.txt"), "w") as f:
    f.write(objectives)

with open(os.path.join(WORKING_DIR, "warnings.txt"), "w") as f:
    for line in warnings:
        f.write(line)



RTP = RealTimeVideoProcess(information_processor=INFORMATION_PROCESSOR, 
                           timestamp_extractor=TIMESTAMP_EXTRACTOR, 
                           video_question_generator=VIDEO_QUESTION_GENERATOR)

log.info("RTP Loaded")

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
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save file locally in 'uploads' folder (make sure folder exists)
        upload_folder = WORKING_DIR
        
        video_path = os.path.join(upload_folder, "original_video", filename)
        file.save(video_path)

        # Create an output folder for chunked videos/text
        output_dir = os.path.join(upload_folder, "generated_video_content")
        os.makedirs(output_dir, exist_ok=True)

        # Run the real_time_video_process in a new Thread
        def background_task():
            try:
                RTP.real_time_video_process(video_path, output_dir)
            except Exception as e:
                print(f"Background task error: {e}")

        thread = threading.Thread(target=background_task, daemon=True)
        thread.start()

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

    # For demonstration, we just return the top-1 match
    results = INFORMATION_PROCESSOR.query(question, top_k=1)
    if results:
        # Return the text from the best match
        return jsonify({"answer": results[0]['text']})
    else:
        return jsonify({"answer": "No data in vector DB yet."})

# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == '__main__':
    app.run(debug=True)
