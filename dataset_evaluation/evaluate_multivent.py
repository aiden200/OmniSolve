from dataset_evaluation.rag_system import RAGSystem
import os, csv
from tqdm import tqdm






def evaluate(multivent_video_path, multivent_path, summarization_eval=False, filters="english", store_name="multivent_summary"):

    rag_system = RAGSystem()
    rag_system.read_video_llm_file(multivent_video_path, summarization_eval, store_name, filters=filters)


    if rag_system.db == None:
        raise ValueError("RAG has not yet been populated")
    
    total = 0
    top_1 = 0
    top_5 = 0
    top_10 = 0


    with open(os.path.join(multivent_path, "multivent_base.csv")) as csvfile:
        spamreader = csv.reader(csvfile)
        next(spamreader)
        i = 0
        for row in tqdm(spamreader):
            video_URL,video_description,language,event_category,event_name,article_url,en_article_url,en_article_excerpt = row
            
            if "youtube.com" in video_URL: 
                # extract the back end
                unique_name = video_URL.split("youtube.com/watch?v=")[-1]
            elif "twitter.com" in video_URL:
                unique_name = video_URL.split("http://twitter.com/i/status/")[-1]

            if filters and (filters == "english" and language != "english"):
                continue
            if unique_name in rag_system.failed:
                continue
            
            
            if not os.path.exists(os.path.join(multivent_video_path, "split_videos", unique_name)):
                continue
            total += 1

            relevant_docs = rag_system.query_vector_store(video_description, store_name)


            doc_ids = []
            for doc in relevant_docs:
                doc_ids.append(doc.metadata["full_video_id"])
            
            # print(video_description)
            # print(relevant_docs[0].page_content)
            # print(unique_name, doc_ids)

            if len(doc_ids) >= 1:
                if doc_ids[0] == unique_name:
                    # print(video_description)
                    # print(relevant_docs[0].page_content)
                    # print(unique_name, doc_ids)
                    top_1 += 1
                if unique_name in doc_ids[:5]:
                    top_5 += 1
                if unique_name in doc_ids[:10]:
                    top_10 += 1
            # print(top_1, top_5, top_10, total)
            
            # i += 1
            # if i == 3:
            #     exit(0)
    print(f"Results\nR@1: {top_1/total}\nR@5: {top_5/total}\nR@10: {top_10/total}")
            

if __name__ == "__main__":
    evaluate("multivent_eval", "/home/aiden/Documents/cs/multiVENT/data")





