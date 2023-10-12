
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

def transcribe_whisper(model):
    output = model.transcribe("youtube_video.mp4", verbose=False)

    transcript = output['text']

    return transcript

def check_subtitles(yt_video):
    try:
        subs = yt_video.captions['a.en']
        srt_subs = subs.generate_srt_captions()
        return srt_subs
    except Exception as e:
        print(e)
        return False

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
map_prompt_template = PromptTemplate(template=MAP_PROMPT, input_variables=["text"])


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
combine_prompt_template = PromptTemplate(template=COMBINE_PROMPT, input_variables=["text"])

def generate_summary(text, llm):
    summary = ""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)

    docs = text_splitter.create_documents([text])
    num_docs = len(docs)
    num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)
    print(f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens")
    
    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                    )
    
    summary = summary_chain.run(docs)

    return summary

def get_embeddings():
    raise NotImplementedError
