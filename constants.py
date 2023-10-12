
MAP_PROMPT = """
Write a concise summary of the following:
"{text}"
Summarise according to the following criteria:
- Concise
- Accurate
- Relevant
- No plagiarism
- No quotes
CONCISE SUMMARY:
"""

COMBINE_PROMPT = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
Summarise according to the following criteria:
- Concise
- Accurate
- Relevant
- No plagiarism
- No quotes
BULLET POINT SUMMARY:
"""
