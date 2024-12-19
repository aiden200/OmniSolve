# Multimodal Evidence Question Generation System

## Overview  
This repository provides a next-generation **Multimodal Question Generation** system that anticipates user queries across diverse evidence sources, including **text, images, and video**. The system employs advanced methods to generate **multi-hop questions** and **multimodal reasoning**, ensuring a more holistic understanding and alignment with user search intent.

Unlike traditional systems, this approach compares answers using **BERT-based similarity embeddings** for accuracy and enables seamless **multi-modal Q&A**, including video inputs.

---

## Key Features  

### 1. **Multimodal Question Generation**  
Our system handles multiple evidence types:  
- **Text**: Generates precise, contextual questions answerable solely from the text.  
- **Images**: Anticipates event-driven visual questions, ensuring the focus is on evidence-based reasoning.  
- **Videos**: Extracts temporal insights to generate time-sensitive queries and answers.  

---

### 2. **Multi-Hop Reasoning**  
Supports multi-hop questions requiring reasoning across multiple evidence sources or modalities. For instance:  
- Given both text and an image, generate questions that integrate context from both.  
- Link temporal spans in videos to supporting text-based metadata for cross-modal queries.  

**Example**:  
*"How many vehicles were damaged during the explosion, and what kind of emergency equipment was used for rescue?"*  

---

### 3. **Answer Comparison with BERT Embeddings**  
We evaluate and compare the **semantic similarity** of answers using **BERT-based embeddings**. This ensures accurate alignment of generated answers, even when phrased differently.

**Benefits**:  
- Tolerance to paraphrasing.  
- Better matching of complex answers across modalities.

---

## System Design  

### Core Components  
1. **Textual Question Generator**  
   - Tokenizes content into sentences/paragraphs.  
   - Generates JSON-formatted question-answer pairs using textual context.  

2. **Visual Question Generator**  
   - Processes images to derive event-centric questions.  
   - Integrates **image metadata** (captions, alt text, descriptions) for additional insights.  

3. **Video Question Generator**  
   - Handles temporal segmentation.  
   - Generates time-stamped Q&A pairs for critical video frames.  

4. **Multi-Hop Question Engine**  
   - Combines insights across multiple evidence types (text, image, video) for coherent multi-hop questions.  

5. **Answer Similarity Scorer**  
   - Computes embedding similarity between generated answers using **BERT-based models**.  

---

## Setup  

### Prerequisites  
- Python 3.9+  
- Required libraries:  
```bash
pip install anthropic sentence-transformers nltk opencv-python-headless
```

### Environment Configuration
Set up an environment variable for your API key:
```bash
export CLAUDE_API_KEY=your-api-key
export GOOGLE_API_KEY=your_api_key_here
```

---

---
## Usage

### 1. **Generating Textual Evidence:**  
```python
from question_generator import QuestionGenerator, TextualEvidence

generator = QuestionGenerator(api_key="your-api-key")
evidence = TextualEvidence(id="e1", content="Your text content here", title="Sample Title")
questions = generator.process_textual_source(evidence)

print(questions)
```

### 2. **Generating Visual Evidence:**  
```python
from question_generator import VisualEvidence

visual_evidence = VisualEvidence(
    id="e2", 
    image_path="path/to/image.jpg", 
    metadata={"human-written caption": "A large fire destroyed the building."}
)
questions = generator.process_visual_source(visual_evidence)

print(questions)

```

### 3. **Generating Video Evidence (using the full video):**  
```python
from video_question_generator import VideoQuestionGenerator

generator = VideoQuestionGenerator()

video_path = "path/to/video.mp4"
response = generator.qa_over_entire_video(video_path)

print(response)  # Outputs generated questions and text summaries for the entire video.


```

### 4. **Generating Video Evidence (Processing Video Segments by Timestamp):**  
```python
from video_question_generator import VideoQuestionGenerator

generator = VideoQuestionGenerator()

video_path = "path/to/video.mp4"
start_time = 10  # Start timestamp in seconds
end_time = 20    # End timestamp in seconds

vid_output_path = "path/to/output_segment.mp4"
qa_output_path = "path/to/questions.json"
text_output_path = "path/to/summary.txt"

generator.qa_over_part_video(
    video_path,
    start_time,
    end_time,
    vid_output_path,
    qa_output_path,
    text_output_path
)

print(f"Segment saved to {vid_output_path}")
print(f"Questions saved to {qa_output_path}")
print(f"Summary saved to {text_output_path}")
```
---


---

## Output Format  

### Textual Evidence  

```json
[
    {
        "question": "What time did the derailment occur?",
        "context_span": "9:15 PM"
    }
]
```

### Visual Evidence  

```json
[
    {
        "question": "What happened at the start of the video?",
        "answer": "A train derailment occurred near the station."
    },
    {
        "question": "What were the immediate consequences?",
        "answer": "Several cars were overturned, spilling their contents."
    }
]


```
### Multi-Hop Example  

```json
{
    "question": "How many cars were damaged, and what equipment was used for recovery?",
    "answers": [
        {"source": "text", "answer": "9 coal cars"},
        {"source": "image", "answer": "Heavy-duty cranes and trucks"}
    ]
}

```
---