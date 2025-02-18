import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import subprocess


def get_video_info_opencv(video_path):
    """
    Returns (fps, duration_in_seconds) for the given video_path using OpenCV.
    WARNING: If the video is variable frame rate (VFR) or metadata is inaccurate,
    this may be off. Use ffprobe for more reliable data if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0.0
    
    return fps, duration

def get_video_fps(video_path):
    video_capture = cv2.VideoCapture(video_path)
    # duration = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    return fps

def get_video_duration(video_path):
    video_capture = cv2.VideoCapture(video_path)
    duration = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    # fps = video_capture.get(cv2.CAP_PROP_FPS)

    return duration



def extract_frame_from_video(video_path, frame_index):
    """
    Opens a video file at `video_path` with OpenCV,
    seeks to `frame_index`, grabs that frame, and returns as a numpy array (BGR).
    Returns None if reading fails.
    """

    curr_frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    while ret:
        if frame_index == curr_frame_idx:
            break
        ret, frame = cap.read()
        curr_frame_idx += 1
    
    cap.release()
    
    return frame, fps

    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return None
    
    # Ask OpenCV to seek directly to frame_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    ret, frame = cap.read()
    cap.release()
    
    # if not ret:
    #     print(f"Error: Could not read frame {frame_index} from {video_path}")
    #     return None
    
    return frame, fps



def find_matching_frame(original_video_path, subclip_video_folder, original_frame_indexes, visualize=True):
    """
    Extract the frame from the original video at the given frame index, then
    iterate through all frames in the subclip video to find the frame that matches exactly.
    
    :param original_video_path: Path to the original video file.
    :param subclip_video_path: Path to the subclip video file.
    :param original_frame_index: The frame index in the original video to match.
    :return: A tuple (matching_subclip_frame_index, matching_frame) if found, otherwise (None, None).
    """


    index_of_frames = 0
    current_info = {}
    if not original_frame_indexes:
        # print(original_video_path, original_frame_indexes)
        # exit(0)
        return 1, current_info
    target_frame= extract_frame_from_video(original_video_path, original_frame_indexes[index_of_frames])[0]
    # try:
    #     target_frame = extract_frame_from_video(original_video_path, original_frame_indexes[index_of_frames])[0]
    # except Exception as e:
    #     print(f"Video {original_video_path} with exception {e}")
    #     return -1, None

    # Extract the target frame from the original video.
    i = 0
    while os.path.exists(os.path.join(subclip_video_folder, f"{i}.mp4")):
        subclip_video_path = os.path.join(subclip_video_folder, f"{i}.mp4")
    
        cap = cv2.VideoCapture(subclip_video_path)
        subclip_frame_index = 0
        
        while True:
            
            ret, frame = cap.read()

            if not ret:
                break  # End of subclip video
            
            # Compare the current subclip frame with the target frame.
            # np.array_equal returns True if both arrays have the same shape and elements.
            if np.array_equal(target_frame, frame):
                current_info[original_frame_indexes[index_of_frames]] = [subclip_video_path, subclip_frame_index]
                index_of_frames += 1
                if index_of_frames == len(original_frame_indexes):
                    return 1, current_info
                
                if visualize:
                    # For a simple pixel-wise check (BGR difference):

                    original_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                    subclip_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(original_frame_rgb)
                    axes[0].set_title("Original Video Frame")
                    axes[0].axis("off")

                    axes[1].imshow(subclip_frame_rgb)
                    axes[1].set_title("Extracted Frame from Subclip")
                    axes[1].axis("off")
                    plt.show()

                target_frame = extract_frame_from_video(original_video_path, original_frame_indexes[index_of_frames])[0]
                


            # print(type(target_frame), type(frame))
            # exit(0)
            # diff = cv2.absdiff(target_frame, frame)
            # print(diff)
            # print(f"Finding frame: {original_frame_indexes[index_of_frames]}, on frame: {subclip_frame_index}, video: {i}")
            subclip_frame_index += 1

        i += 1
        cap.release()

    # TODO: This might need a fix
    if index_of_frames != 0:
        return 1, current_info
    print(index_of_frames, len(original_frame_indexes), original_frame_indexes)

    return -1, None


def get_video_info_opencv(video_path):
    """
    Returns (fps, duration_in_seconds) for the given video_path using OpenCV.
    WARNING: If the video is variable frame rate (VFR) or metadata is inaccurate,
    this may be off. Use ffprobe for more reliable data if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0.0
    
    return fps, duration


def find_subclip_and_local_frame(original_frame_idx, subclip_folder, fps_orig):
    """
    Given an original frame index (0-based) in the original video,
    returns (subclip_folder, local_frame_index) for the correct subclip,
    or (None, None) if not found.
    """
    time_in_seconds = original_frame_idx / fps_orig

    results = []
    i = 0

    start_time = 0

    while os.path.exists(os.path.join(subclip_folder, f"{i}.mp4")):
        subclip_path = os.path.join(subclip_folder, f"{i}.mp4")
        subclip_fps, clip_duration = get_video_info_opencv(subclip_path)
        
        results.append({
            "path": subclip_path,
            "start": start_time,
            "end": start_time + clip_duration,
            "fps": subclip_fps
            # "fps": fps_orig
        })

        start_time = start_time + clip_duration + 1
        i += 1
    
    # print(results)
    
    # Find which subclip interval covers this time
    for clip_info in results:
        start_sec = clip_info["start"]
        end_sec = clip_info["end"]
        if start_sec <= time_in_seconds < end_sec:
            # Found subclip that contains this frame
            subclip_path = clip_info["path"]
            fps_subclip = clip_info["fps"]
            # local subclip frame = offset in seconds * subclip fps
            local_frame_idx = int((time_in_seconds - start_sec) * fps_subclip)
            return subclip_path, local_frame_idx
    
    # If no subclip found, we return None
    print(f"Warning: No subclip covers time={time_in_seconds:.2f}s.")
    return None, None

########################################################
# 4. MAIN FUNCTION: EXTRACT CORRECT FRAME FROM SUBCLIPS
########################################################

def extract_frame_from_subclips(original_video_path, subclip_path, fps_orig, original_frame_idx):
    """
    Finds which subclip covers the frame, then extracts the correct
    frame from that subclip.
    """
    subclip_path, local_frame_idx = find_subclip_and_local_frame(original_frame_idx, subclip_path, fps_orig)
    if subclip_path is None:
        return None
    
    # Now actually extract from subclip
    frame, fps = extract_frame_from_video(subclip_path, local_frame_idx)
    return frame, subclip_path



########################################################
# 5. TEST / SANITY CHECK
########################################################

def test_frame_extraction(original_video_path, subclip_path, original_frame_idx):
    """
    1. Extract the frame (original_frame_idx) directly from the original video.
    2. Extract the same frame via the subclip approach.
    3. Compare if they match.
    """
    print(f"\n=== Testing frame extraction for original frame index {original_frame_idx} ===")

    
    
    # (A) Extract directly from original
    original_frame, fps_orig = extract_frame_from_video(original_video_path, original_frame_idx)
    if original_frame is None:
        print("Failed to extract from original video.")
        return


    # (B) Extract via subclip approach
    subclip_frame, subclip_path = extract_frame_from_subclips(original_video_path, subclip_path, fps_orig, original_frame_idx)
    if subclip_frame is None:
        print("Failed to extract via subclip. Possibly out of range.")
        return
    
    # Compare shape
    if original_frame.shape != subclip_frame.shape:
        print(f"Frame shape mismatch: original={original_frame.shape}, subclip={subclip_frame.shape}")
        return
    
    # For a simple pixel-wise check (BGR difference):
    diff = cv2.absdiff(original_frame, subclip_frame)
    num_diff = np.sum(diff)

    original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    subclip_frame_rgb = cv2.cvtColor(subclip_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_frame_rgb)
    axes[0].set_title("Original Video Frame")
    axes[0].axis("off")

    axes[1].imshow(subclip_frame_rgb)
    axes[1].set_title("Extracted Frame from Subclip")
    axes[1].axis("off")

    print(f"Subclip in question: {subclip_path}")

    if num_diff == 0:
        print("✅ Frames match perfectly!")
    else:
        print(f"⚠️ Frames differ by pixel difference sum = {num_diff}")

        # Display heatmap of differences
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        axes[2].imshow(diff_gray, cmap="hot")
        axes[2].set_title("Difference Heatmap")
        axes[2].axis("off")
    if num_diff == 0:
        print("Frames match perfectly!")
    else:
        # Some differences might exist due to re-encoding, but hopefully minimal
        print(f"Frames differ by sum of pixel abs differences = {num_diff}")
        # You could implement a threshold check if needed.
    plt.show()



def multivent_g_find_frames(multivent_g_path, subclip_path, original_video_path, savefile_path):
    if os.path.exists(savefile_path):
        with open(savefile_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    with open(multivent_g_path, "r") as f:
        gt_data = json.load(f)
    
    count = 0
    
    for video in tqdm(gt_data):
        # try:
        video_subclip_path = os.path.join(subclip_path, video)
        video_path = os.path.join(original_video_path, f"{video}.mp4")
        if os.path.exists(video_subclip_path) and video not in data:
            frame_idxs = []
            for object in gt_data[video]["spatial"]:
                if object["frame"] not in frame_idxs:
                    frame_idxs.append(object["frame"])
            
            frame_idxs.sort()
            try:
                status, frame_info = find_matching_frame(video_path, video_subclip_path, frame_idxs, visualize=False)
            except Exception:
                # Convert frame back into a good one
                new_video_path = os.path.join(original_video_path, f"{video}_converted.mp4")
                print(new_video_path)
                cmd = [
                    "ffmpeg",
                    "-y",               # Overwrite output file if it exists
                    "-i", video_path,   # Input file
                    "-c:v", "libx264",  # Use H.264 codec for video
                    "-c:a", "copy",     # Copy audio stream without re-encoding (optional)
                    new_video_path
                ]

                subprocess.run(cmd, check=True)
                status, frame_info = find_matching_frame(new_video_path, video_subclip_path, frame_idxs, visualize=False)
            # print(status, frame_info)
            print(video, video_path)
            if status == 1:
                data[video] = frame_info

                with open(savefile_path, "w") as f:
                    json.dump(data, f, indent=2)
            count += 1
        elif video in data:
            # print(f"video {video} already loaded")
            count += 1
        # except Exception as e:
        #     print(e)
    print(f"Processed: {len(data)} frame conversion data. Total: {count}")


def check_counter_of_files_multivent_g(subclip_path):
    finished_files = set()
    status_file = os.path.join(subclip_path, "status.txt")
    with open(status_file, "r") as f:
        files = f.readlines()
        for names in files:
            name = names.split("/")[3]
            finished_files.add(name)

    print(f"Processed: {len(finished_files)} clips and {len(files)} clips")





if __name__ == "__main__":
    # check_counter_of_files_multivent_g("/data/multivent_processed_without_delay")
    multivent_g_path = "/home/aiden/Documents/cs/multiVENT/data/multivent_g.json"
    subclip_path = "/data/multivent_processed_without_delay"
    original_video_path = "/data/multivent_yt_videos"
    savefile_path = os.path.join(subclip_path, "frame_convert.json")
    multivent_g_find_frames(multivent_g_path, subclip_path, original_video_path, savefile_path)
    # # Example: we want to check frames #100, #223 from the original
    # frames_to_test = [110, 122, 200, 300, 400, 490, 800]
    # # frames_to_test = [1000]

    # vid_name = "Dk1y6G7hhUo"
    # original_video_path = f"/data/multivent_yt_videos/{vid_name}.mp4"
    # subclip_path = f"/data/multivent_processed/{vid_name}"
    # # for f_idx in frames_to_test:
    # #     test_frame_extraction(original_video_path, subclip_path, f_idx)
    

    # find_matching_frame(vid_name, original_video_path, subclip_path, frames_to_test, visualize=True)

