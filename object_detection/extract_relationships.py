

from dotenv import load_dotenv
from google import genai
import os
import time
import json 
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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


class RelationshipExtractor:
    def __init__(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.url = os.environ["NEO4J_URI"]
        self.username = os.environ["NEO4J_USERNAME"]
        self.password = os.environ["NEO4J_PASSWORD"]


    def build_relationships(self, relationships, video, subclip_id, start, end, existing=False):

        docs = []
        for rel in relationships:
            # Assume each of object1 and object2 are lists; take first element as the representative name.
            obj1 = rel.get("object1", [""])[0]
            rel_type = rel.get("relationship", "")
            obj2 = rel.get("object2", [""])[0]
            # Create a human-readable description of the relationship.
            description = f"{obj1} {rel_type} {obj2}"
            # Include temporal metadata (e.g., the subclip identifier).
            # metadata = {"subclip_id": subclip_id}
            metadata = {"subclip_id": subclip_id, "start_time": start, "end_time": end}
            doc = Document(page_content=description, metadata=metadata)
            docs.append(doc)

        index_name = f"{video}_relationships_{subclip_id}"

        if existing:
            store = Neo4jVector.from_existing_index(
                self.hf_embeddings,
                url=self.url,
                username=self.username,
                password=self.password,
                index_name=index_name,
            )
        else:
            store = Neo4jVector.from_documents(
                docs, 
                self.hf_embeddings, 
                url=self.url, 
                username=self.username, 
                password=self.password,
                index_name=index_name
            )
        print(f"Stored {len(docs)} relationship documents in index '{index_name}'.")
        return store


    def query_hybrid_system(query_text, query_start_time, query_end_time, index_name):
        # Step 1: Semantic Query on the Vector Database
        # (Assume vector_db.search returns a list of subclip metadata with subclip_id, start_time, end_time)
        candidate_subclips = vector_db.search(query_text)
        
        # Filter by time overlap (if vector DB doesn’t already do this)
        candidate_subclip_ids = []
        for subclip in candidate_subclips:
            if subclip['start_time'] <= query_end_time and subclip['end_time'] >= query_start_time:
                candidate_subclip_ids.append(subclip['subclip_id'])
        
        # Step 2: Temporal Query on the KG
        # Compose a Cypher query with time and candidate subclip filtering.
        cypher_query = """
        MATCH (a)-[r]->(b)
        WHERE r.subclip_id IN $candidate_ids
        AND r.start_time <= $query_end_time
        AND r.end_time >= $query_start_time
        // Optionally add semantic conditions:
        AND (a.name CONTAINS $keyword OR b.name CONTAINS $keyword)
        RETURN a, r, b
        """
        # You could extract keywords from the query_text; for simplicity, we pass one keyword here.
        params = {
            "candidate_ids": candidate_subclip_ids,
            "query_start_time": query_start_time,
            "query_end_time": query_end_time,
            "keyword": "fireman"  # or extract dynamically from query_text
        }
        kg_results = neo4j_driver.run(cypher_query, params)

        # Combine and rank results as needed
        final_results = process_results(candidate_subclips, kg_results)
        return final_results



def extract_relationship_for_video(video_folder, client, prompt, debug=False):

    i = 0
    while os.path.exists(os.path.join(video_folder, f"{i}_RGS.mp4")):
        detection_file = os.path.join(video_folder, f"{i}_object_relationships.json")
        if os.path.exists(detection_file):
            if debug:
                print(f"File {detection_file} already exists")
            i += 1
            continue
        if debug:
            print(f"On file {detection_file}")
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
        # if debug:
        #     print("object")

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
        extract_relationship_for_video(video, client, prompt, debug)


if __name__ == "__main__":
    debug=True
    multivent_processed_folder = "/data/multivent_processed_without_delay/"
    extract_relationships(multivent_processed_folder,debug=debug)


    
        
