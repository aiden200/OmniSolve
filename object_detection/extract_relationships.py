

from dotenv import load_dotenv
from google import genai
import os
import time
import json 
from tqdm import tqdm
load_dotenv()



def extract_json_between_markers(text, start_marker="```json", end_marker="```"):
        try:
            # Find the start and end positions
            start_pos = text.index(start_marker) + len(start_marker)
            end_pos = text.index(end_marker, start_pos)
            
            # Extract and return the text between the markers
            text = text[start_pos:end_pos].strip()
            json_data = json.loads(text)
            return json_data
        except ValueError:
            print("Markers not found in the text.")
            return None

def extract_relationship_for_video(video_folder, client, prompt):

    i = 0
    while os.path.exists(os.path.join(video_folder, f"{i}_RGS.mp4")):
        detection_file = os.path.join(video_folder, f"{i}_object_relationships.json")
        if os.path.exists(detection_file):
            continue
        object_folder = os.path.join(video_folder, f"{i}_detection_results", "json_data")
        object_files = [f for f in os.listdir(object_folder)]
        object_files.sort()
        # We only need to detect the final json file, contains all objects
        object_file = os.path.join(object_folder, object_files[-1])
        with open(object_file, "r") as f:
            all_object_info = json.load(f)
        
        object_names = {}
        for instance_id in all_object_info["labels"]:
            object_names[int(instance_id)] = all_object_info['labels'][instance_id]['class_name']
        
        final_instance_id = int(instance_id)
        all_instance_ids = list(range(1, final_instance_id + 1))
        
        # We need all the objects
        for f in object_files:
            object_file = os.path.join(object_folder, f)
            with open(object_file, "r") as f:
                all_object_info = json.load(f)
            
            instance_keys = sorted(list(object_names.keys()))
            if instance_keys== all_instance_ids:
                break

            for instance_id in all_object_info["labels"]:
                if int(instance_id) not in object_names:
                    object_names[int(instance_id)] = all_object_info['labels'][instance_id]['class_name']

        object_names_str = ""
        for instance_id in instance_keys:
            object_names_str = f"{object_names[instance_id]}, ID: {str(instance_id)}\n" + object_names_str


        detected_video_file = os.path.join(video_folder, f"{i}_RGS.mp4")
        video_file = client.files.upload(file=detected_video_file)

        # print("starting process")

        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)



        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=[
                video_file,
                f"Objects:\n{object_names_str} Given the highlighted objects of the video, return the relationships \
                    between the objects in a json format. Return Only unique relationships\
                        Example Return Format:{prompt}"])

        i += 1
        json_response = extract_json_between_markers(response.text)
        with open(detection_file, "w") as f:
            json.dump(json_response, f)


def extract_relationships(multivent_processed_folder, debug=False):

    prompt = '''
    [
        {
            "object1": [OBJECT 1 NAME],
            "relationship": "is behind",
            "object2": [OBJECT 2 NAME],
        },
        ...
    ]
    '''

    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    videos = [f.path for f in os.scandir(multivent_processed_folder) if f.is_dir()]

    for video in tqdm(videos):
        if debug:
            print(video)
        extract_relationship_for_video(video, client, prompt)


if __name__ == "__main__":
    debug=True
    multivent_processed_folder = "/data/multivent_processed_without_delay/"
    extract_relationships(multivent_processed_folder,debug=debug)


    
        
