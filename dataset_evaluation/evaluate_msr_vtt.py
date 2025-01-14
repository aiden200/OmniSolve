from datasets import load_dataset
import yt_dlp
from tqdm import tqdm
import os, csv
import shutil



def dwl_vid(video_url, save_path, cookie_path, filename):
    
    if os.path.exists(f"{save_path}/{filename}.mp4"):
        print("Skipping, yt")
        return
    
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Combine the best video and audio streams
        'outtmpl': f'{save_path}/{filename}.%(ext)s',       # Custom filename template
        'cookiefile': f"{cookie_path}",
        "cachedir": False,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',    # Convert video format
            'preferedformat': 'mp4',          # Convert to MP4
        }],
        'quiet': True,             # Suppresses all output messages
        'no_warnings': True,  
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def download_msr_vtt_dataset(msr_vtt_save_path, cookie_path, additional_paths = None):
    
    if not os.path.exists(msr_vtt_save_path):
        os.mkdir(msr_vtt_save_path)
    
    write_file = os.path.join(msr_vtt_path, "results.csv")
    
    video_save_path = os.path.join(msr_vtt_save_path, "videos")

    # dataset = load_dataset("AlexZigma/msr-vtt")
    train_dataset = load_dataset("AlexZigma/msr-vtt", split="train")
    valid_dataset = load_dataset("AlexZigma/msr-vtt", split="val")

    dataset = train_dataset["url"] + valid_dataset["url"]

    with open(write_file, "w", newline="") as f:
        writer = csv.writer(f)

    for vid in tqdm(dataset):
        unique_name = vid.split("youtube.com/watch?v=")[-1]
        try:
            destination_path = os.path.join(video_save_path, f"{unique_name}.mp4")

            if not os.path.exists(destination_path):
                if additional_paths:
                    for folder in additional_paths:
                        check_path = os.path.join(folder, f"{unique_name}.mp4")
                        if os.path.exists(check_path):
                            shutil.copy(check_path, destination_path)
            
            if not os.path.exists(destination_path):
                dwl_vid(vid, destination_path, cookie_path, f"{unique_name}.mp4")
            
            with open(write_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([unique_name, destination_path, False]) # last col is failed
        except Exception as e:
            with open(write_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([unique_name, destination_path, True])

    


def load_msr_vtt_dataset():
    dataset = load_dataset("AlexZigma/msr-vtt")

    # or load the separate splits if the dataset has train/validation/test splits
    train_dataset = load_dataset("AlexZigma/msr-vtt", split="train")
    valid_dataset = load_dataset("AlexZigma/msr-vtt", split="val")
    # test_dataset  = load_dataset("AlexZigma/msr-vtt", split="test")
    print(train_dataset, valid_dataset )
    print(valid_dataset["url"])

# load_msr_vtt_dataset()
cookie_path = "/home/aiden/Documents/cs/multiVENT/data/yt_cookies.txt"
msr_vtt_path = "/home/aiden/Documents/cs/multiVENT/msr_vtt"
additional_paths = ["/home/aiden/Documents/cs/multiVENT/multivent1", "/home/aiden/Documents/cs/multiVENT/yt_videos"]
download_msr_vtt_dataset(msr_vtt_path, cookie_path, additional_paths=additional_paths)