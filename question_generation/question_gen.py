import uuid
import base64
from pathlib import Path
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from typing import List
from anthropic import Anthropic
import json
import logging
from sentence_transformers import SentenceTransformer
from .models import (
    QuestionAnswer,
    TextualEvidence,
    VisualEvidence,
    EvidenceType
)

logger = logging.getLogger(__name__)

TEXTUAL_SYSTEM_PROMPT = """You are helping build a search system that matches user questions to specific pieces of evidence. Generate UP TO 10 questions where this exact text would be the perfect answer.

Requirements:
1. Generate no more than 10 questions
2. Each question MUST be completely answerable using ONLY the provided text chunk - do not generate questions that require any external information
3. Generate only questions where the complete answer is contained within this specific text
4. Questions should reflect how real users would ask

Example text:
"The derailment occurred at 9:15 PM when nine coal cars left the tracks. No hazardous materials were involved."

Good questions (because answers are completely contained in text):
- What time did the derailment occur?
- How many coal cars derailed?
- Were hazardous materials involved in the derailment?

Bad questions (because they need external information):
- Where did the derailment occur? (location not in text)
- What caused the derailment? (cause not in text)
- How long did cleanup take? (not mentioned in text)

Format your response as a JSON array of question-answer pairs:
{
    "qa_pairs": [
        {
            "question": "What time did the derailment occur?",
            "context_span": "9:15 PM"
        }
    ]
}"""

VISUAL_SYSTEM_PROMPT = """You are helping build a news search system that matches user questions to relevant evidence. Your task is to generate questions that could be answered by news imagery - these are questions users might ask about newsworthy events before seeing any images.

Key Understanding:
- Users will ask questions about events/situations, not about specific images
- Images serve as evidence to answer their questions
- Questions should capture what users want to know that visual evidence could answer

Requirements:
1. Generate no more than 10 questions
2. Generate questions that visual evidence could answer about:
  - Physical impact and extent of damage
  - Scale of emergency response
  - Size of crowds or affected areas
  - Visual proof of reported situations
  - etc.

2. Questions must be:
  - Answerable through visual evidence
  - Written as event/situation questions, not image-specific questions
  - In natural and domain adaptable language/terminology
  - Focused on newsworthy aspects

Good questions:
- "How badly was the infrastructure damaged?"  (user asking about event, image shows damage)
- "What type of rescue equipment was used?"  (user asking about response, image shows equipment)
- "How many buildings were affected?" (user asking about impact, image shows scope)

Bad questions:
- "What can be seen in this photo?" (about the image itself)
- "Is this an aerial view?" (about photography)
- "What angle was this taken from?" (about image creation)
- "Can you describe the scene?" (asking for image description)

Format your response as a JSON array of questions:
{
    "questions": [
        "question1",
        "question2"
    ]
}"""

class QuestionGenerator:
    """Anticipates and generates questions that match real user query patterns"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize tokenizer directly
        self.sentence_tokenizer = PunktSentenceTokenizer()
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Wrapper for sentence tokenization to ensure consistent behavior"""
        return self.sentence_tokenizer.tokenize(text)

    def _generate_questions_for_text(self, text: str, context: str = "") -> List[dict]:
        """Generate precise questions that match text chunks"""
        system_message = TEXTUAL_SYSTEM_PROMPT

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Generate questions that can be answered solely from this text{' with the following context' if context else ''}:
                        
                        {f'Context: {context}' if context else ''}
                        Text: {text}

                        Remember: 
                        1. Every question must be completely answerable using ONLY this specific text
                        2. For each question, provide the exact span of text that contains the answer
                        3. The span should be the minimal text needed to answer the question"""
                    }
                ]
            )
            
            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                qa_pairs = json.loads(json_str)["qa_pairs"]
                return qa_pairs
            
            logger.warning(f"Could not find valid JSON in response: {response_text}")
            return []
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _generate_questions_for_image(self, image_path: Path) -> List[str]:
        """Generate questions about the visual content of an image"""
        try:
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")

            # Determine media type based on file extension
            media_type = f"image/{image_path.suffix[1:].lower()}"  

            system_message = VISUAL_SYSTEM_PROMPT

            response = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": """Analyze this news image and generate questions that users would ask about this event BEFORE seeing any imagery. 

Requirements:
1. Questions must be about the EVENT/SITUATION, not about the image itself
2. BAD questions (do not generate these):
   - "What does the image show?"
   - "Was this taken from above?"
   - "Is this an aerial photograph?"
   - "Who took this photo?"
   
3. GOOD questions (generate these types):
   - "How extensive was the damage?"
   - "What kind of emergency response was deployed?"
   - "How many buildings/structures were affected?"
   - "What environmental impacts resulted from this incident?"

Remember: Users will be asking about the incident BEFORE they see any images. Your questions should reflect what they want to know about the event itself, with the image serving as evidence to answer their questions."""
                            }
                        ],
                    }
                ]
            )
            
            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)["questions"]
            
            return []

        except Exception as e:
            logger.error(f"Error generating questions for image {image_path}: {str(e)}")
            return []

    def process_textual_source(self, evidence: TextualEvidence) -> List[QuestionAnswer]:
        try:
            all_questions = []
            # Split into paragraphs first
            paragraphs = evidence.content.split('\n\n')
            
            for para_num, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                    
                sentences = self._split_into_sentences(paragraph)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    # Check if adding this sentence would exceed chunk size
                    if current_length + len(sentence) > 500:
                        # Process current chunk before starting new one
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            qa_pairs = self._generate_questions_for_text(
                                chunk_text,
                                f"From paragraph {para_num + 1} of article: {evidence.title}"
                            )
                            
                            if qa_pairs:
                                all_questions.extend([
                                    QuestionAnswer(
                                        question_id=f"q_{uuid.uuid4().hex}",
                                        question_text=pair['question'],
                                        evidence_id=evidence.id,
                                        evidence_type=EvidenceType.TEXTUAL,
                                        source_text=chunk_text,
                                        source_type="article_chunk",
                                        context_span=pair['context_span']
                                    )
                                    for pair in qa_pairs
                                ])
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
            
                # Process final chunk if it exists
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    qa_pairs = self._generate_questions_for_text(
                        chunk_text,
                        f"From paragraph {para_num + 1} of article: {evidence.title}"
                    )
                    
                    if qa_pairs:
                        all_questions.extend([
                            QuestionAnswer(
                                question_id=f"q_{uuid.uuid4().hex}",
                                question_text=pair['question'],
                                evidence_id=evidence.id,
                                evidence_type=EvidenceType.TEXTUAL,
                                source_text=chunk_text,
                                source_type="article_chunk",
                                context_span=pair['context_span']
                            )
                            for pair in qa_pairs
                        ])
            
            return all_questions
            
        except Exception as e:
            logger.error(f"Error processing textual evidence {evidence.id}: {str(e)}")
            raise

    def process_visual_source(self, evidence: VisualEvidence) -> List[QuestionAnswer]:
        """Generate questions for both visual content and metadata"""
        all_questions = []
        
        try:
            # First, generate questions about the visual content
            logger.info(f"Analyzing image content for {evidence.id}")
            image_path = Path(evidence.image_path)
            if image_path.exists():
                image_questions = self._generate_questions_for_image(image_path)
                
                if image_questions:
                    all_questions.extend([
                        QuestionAnswer(
                            question_id=f"q_{uuid.uuid4().hex}",
                            question_text=question,
                            evidence_id=evidence.id,
                            evidence_type=EvidenceType.VISUAL,
                            source_text="Visual content of image",
                            source_type="visual",
                        )
                        for question in image_questions
                    ])

            # Process metadata
            metadata_types = {
                'caption': evidence.metadata.get('human-written caption', ''),
                'alt_text': evidence.metadata.get('alt-text', ''),
                'description': evidence.metadata.get('generated-description', ''),
                'text_in_image': evidence.metadata.get('text-in-image', '')
            }

            # Process each type of metadata that exists
            for metadata_type, content in metadata_types.items():
                if not content:
                    continue
                    
                logger.info(f"Processing {metadata_type} for {evidence.id}")
                questions = self._generate_questions_for_text(
                    content,
                    f"This is {metadata_type} describing the image."
                )
                
                if questions:
                    all_questions.extend([
                        QuestionAnswer(
                            question_id=f"q_{uuid.uuid4().hex}",
                            question_text=question,
                            evidence_id=evidence.id,
                            evidence_type=EvidenceType.VISUAL,
                            source_text=content,
                            source_type=metadata_type
                        )
                        for question in questions
                    ])

            logger.info(f"Generated {len(all_questions)} questions for visual evidence {evidence.id}")
            return all_questions

        except Exception as e:
            logger.error(f"Error processing visual evidence {evidence.id}: {str(e)}")
            raise
