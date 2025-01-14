
from generate_questions import VideoParser
from tqdm import tqdm
import csv
import os
import time
from dotenv import load_dotenv
from dataset_evaluation.evaluate_multivent import evaluate
import shutil
import tempfile

load_dotenv()

def evaluate_multivent_1(summarization_eval=False, store_name="multivent_summary"):

    evaluate("multivent_1_eval", "/home/aiden/Documents/cs/multiVENT/data",summarization_eval=summarization_eval, store_name=store_name)


def summarize_videos(multivent_video_path, new_video_path):
    check_path = os.path.join(new_video_path, "split_videos")
    
    if not os.path.exists(multivent_video_path):
        raise ValueError(f"No multivent path {multivent_video_path}")
    
    parser = VideoParser(load_timestamp_processer=False)
    
    output_csv_path = os.path.join(new_video_path, "results.csv")


    with open(output_csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            name, failed, language, event_category, event_name = row
            filename = os.path.join(check_path, name, "summary.txt")
            if failed == "True" or os.path.exists(filename):
                continue
            full_text = ""
            i = 0
            while os.path.exists(os.path.join(check_path, name, f"{i}.txt")):
                with open(os.path.join(check_path, name, f"{i}.txt"), "r") as f:
                    full_text += f.read()
                i += 1
            
            parser.summarize_video_given_text(filename, full_text)
        

def recover_results_file(multivent_video_path, new_video_path,multivent1_clips_path):
    results_path = os.path.join(new_video_path, "results.csv")
    split_videos_path = os.path.join(new_video_path, "split_videos")
    multivent_base_path = os.path.join(multivent_video_path, "multivent_base.csv")

    for root, dirs, files in os.walk(split_videos_path, topdown=False):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                # print(dir_path)
                os.rmdir(dir_path)


    directories = [name for name in os.listdir(split_videos_path)]

    info = {}
    files = [
        name[:-4] for name in os.listdir(multivent1_clips_path)
        if os.path.isfile(os.path.join(multivent1_clips_path, name)) and name.endswith(".mp4")
    ]

    with open(multivent_base_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            video_URL, video_description, language, event_category,event_name, article_url, en_article_url, en_article_excerpt = row
            if "twitter.com" in video_URL:
                unique_name = video_URL.split("twitter.com/i/status/")[-1]
            elif "youtube.com" in video_URL:
                unique_name = video_URL.split("youtube.com/watch?v=")[-1]
            
            info[unique_name] = [unique_name, unique_name not in files, language, event_category, event_name]

    with open(results_path, mode="w", encoding="utf-8") as file:
        writer = csv.writer(file)
        for name in directories:
            # name, failed, language, event_category, event_name
            writer.writerow(info[name])




def write_rest_of_categories(multivent_video_path, new_video_path):
    output_csv_path = os.path.join(new_video_path, "results.csv")
    temp_csv_path = os.path.join(new_video_path, "results_temp.csv")



    values = {}
    with open(os.path.join(multivent_video_path, "multivent_base.csv")) as csvfile:
        spamreader = csv.reader(csvfile)  # Convert to list
        next(spamreader)
        
        for row in tqdm(spamreader):
            video_URL, video_description, language, event_category,event_name, article_url, en_article_url, en_article_excerpt = row
            if "twitter.com" in video_URL:
                unique_name = video_URL.split("twitter.com/i/status/")[-1]
            elif "youtube.com" in video_URL:
                unique_name = video_URL.split("youtube.com/watch?v=")[-1]
            values[unique_name] = [language, event_category, event_name]
    
    # Open the existing CSV file and a temporary file
    with open(output_csv_path, mode="r", encoding="utf-8") as infile, \
         open(temp_csv_path, mode="w", encoding="utf-8", newline="") as tempfile:
        
        # Read from the input file and write to the temporary file
        reader = csv.reader(infile)
        writer = csv.writer(tempfile)

        for row in tqdm(reader):
            
            if len(row) == 2:
                writer.writerow(row + values[row[0]])
            else:
                writer.writerow(row)

    os.replace(temp_csv_path, output_csv_path)


def generate_captions_msr_vtt(msr_vtt, new_video_path):
    split_video_path = os.path.join(new_video_path, "split_videos")
    temporary_dir = os.path.join(msr_vtt, "temp")
    failed_video_path = os.path.join(new_video_path, "failed.csv")
    results_path = os.path.join(msr_vtt, "results.csv")

    if not os.path.exists(new_video_path):
        os.mkdir(new_video_path)
        os.mkdir(split_video_path)
    
    
    if not os.path.exists(msr_vtt):
        raise ValueError(f"No multivent path {msr_vtt}")
    
    if not os.path.exists(temporary_dir):
        os.mkdir(temporary_dir)
    
    parser = VideoParser()
    with open(failed_video_path, "w", newline="") as f:
        writer = csv.writer(f)

    with open(results_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            name, curr_vid_path, failed = row
            if failed == "True":
                with open(failed_video_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, ""]) 
            new_save_path = os.path.join(split_video_path, name)
            # print(failed, failed == "False" and not os.path.exists(new_save_path))
            try:

                if failed == "False" and not os.path.exists(new_save_path):

                    with tempfile.TemporaryDirectory(dir=temporary_dir) as temp_dir:
                        # print(os.path.join(curr_vid_path, f"{name}.mp4"))
                        parser.split_and_store_vids(os.path.join(curr_vid_path, f"{name}.mp4.mp4"), temporary_dir)
                        os.mkdir(new_save_path)
                        shutil.move(temp_dir, new_save_path)

            except Exception as e:
                with open(failed_video_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, e])
                    
                


def generate_captions(multivent_video_path, new_video_path, multivent1_path):
    if not os.path.exists(new_video_path):
        os.mkdir(new_video_path)
        os.mkdir(os.path.join(new_video_path, "split_videos"))
    
    if not os.path.exists(multivent_video_path):
        raise ValueError(f"No multivent path {multivent_video_path}")
    
    finished_videos = []
    parser = VideoParser()
    
    output_csv_path = os.path.join(new_video_path, "results.csv")
    if os.path.exists(output_csv_path):
        with open(output_csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                finished_videos.append(row[0])

    with open(output_csv_path, mode="a", encoding="utf-8") as file:
        writer = csv.writer(file)
        with open(os.path.join(multivent_video_path, "multivent_base.csv")) as csvfile:
            spamreader = csv.reader(csvfile)  # Convert to list
            next(spamreader)
            
            for row in tqdm(spamreader):
                attempt = 0
                while attempt < 1:
                    try:
                        video_URL, video_description, language, event_category,event_name, article_url, en_article_url, en_article_excerpt = row
                        if language != "english":
                            attempt = 6
                            continue
                        if "twitter.com" in video_URL:
                            unique_name = video_URL.split("twitter.com/i/status/")[-1]
                        elif "youtube.com" in video_URL:
                            unique_name = video_URL.split("youtube.com/watch?v=")[-1]
                        

                        if unique_name not in finished_videos:
                            video_store_path = os.path.join(new_video_path, "split_videos", unique_name)
                            if not os.path.exists(video_store_path):
                                os.mkdir(video_store_path)

                            curr_vid_path = os.path.join(multivent1_path, unique_name + ".mp4")
                            if os.path.exists(curr_vid_path):
                                # print(curr_vid_path)
                                parser.split_and_store_vids(curr_vid_path, video_store_path)
                                row = [unique_name, False, language, event_category, event_name]
                            else:
                                row = [unique_name, True, language, event_category, event_name]
                            writer.writerow(row)
                        else:
                            print(f"Skipping {unique_name}")
                        
                        break
                    except Exception as e:
                        time.sleep(30)
                        attempt += 1
                        if attempt >=5:
                            break





# recover_results_file("/home/aiden/Documents/cs/multiVENT/data", "multivent_1_eval", "/home/aiden/Documents/cs/multiVENT/multivent1")
# evaluate_multivent_1(summarization_eval=True, store_name="multivent_summary")
generate_captions_msr_vtt("/home/aiden/Documents/cs/multiVENT/msr_vtt", "msr_vtt_results")
# evaluate_multivent_1(summarization_eval=False, store_name="multivent1")
# summarize_videos("/home/aiden/Documents/cs/multiVENT/data", "multivent_1_eval")
# write_rest_of_categories("/home/aiden/Documents/cs/multiVENT/data", "multivent_1_eval")
# generate_captions("/home/aiden/Documents/cs/multiVENT/data", "multivent_1_eval", "/home/aiden/Documents/cs/multiVENT/multivent1")