VIDEO_PROMPT = """
Generate some questions and answers from the video. Make both broad and detailed questions. Respond in a json format.
Analyze this news video and generate questions that users would ask about this event BEFORE seeing any video. 

Requirements:
1. Questions must be about the EVENT/SITUATION, not about the video itself
2. BAD questions (do not generate these):
   - "What does the video show?"
   - "Was this taken from above?"
   - "Is this an aerial video?"
   - "Who took this video?"
   
3. GOOD questions (generate these types):
   - "How extensive was the damage?"
   - "What kind of emergency response was deployed?"
   - "How many buildings/structures were affected?"
   - "What environmental impacts resulted from this incident?"

Remember: Users will be asking about the incident BEFORE they see any videos. 
Your questions should reflect what they want to know about the event itself, with the video serving as evidence to answer their questions.
Answer in json format
"""


VIDEO_PROMPT2 = """
Create some question answer pairs from the video. Make them broad and detailed,
and only use the context from the video. Make sure to answer in a json format.
"""

DEFAULT_MM_PROMPT = "Describe what is happening in the video"



SUMMARY_VIDEO_PROMPT = "Write a detailed description of what is happening in this video. only respond in regular text, not markdown"

SUMMARY_VIDEO_PROMPT_CONTEXT = "Write a detailed description of what is happening in this video. only respond in regular text, not markdown."