import streamlit as st
import os

from youtube_utils import get_video_info, generate_subtitles
from functions import generate_summary  # , get_answer
import whisper

from langchain.llms import OpenAI

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

import time

process_video = None
process_question = None
go_to_timestamp = None

if 'transcript' not in st.session_state:
    st.session_state.transcript = ""

if 'summary' not in st.session_state:
    st.session_state.summary = ""

st.set_page_config(
    page_title='YouTube Q & A',
    layout='wide',
    initial_sidebar_state='expanded')

model = whisper.load_model('base')
transcript = ""

# Sidebar
with st.sidebar:
    user_secret = st.text_input(label=":green[OpenAI API key]",
                                value="",
                                placeholder="",
                                type="password")

    yt_url = st.text_input(label=":green[Youtube URL]",
                           value="https://www.youtube.com/watch?v=jDsQmbCei7g&t=3s&ab_channel=TheProfGShow%E2%80%93ScottGalloway",
                           placeholder="", type="default")

# if user_secret and yt_url:
    if yt_url:
        if st.button("Start Analysis"):
            if os.path.exists('youtube_video.mp4'):
                os.remove('youtube_video.mp4')
            stream, title = get_video_info(yt_url)
            st.write(title)
            st.video(yt_url)

            audio = open('youtube_video.mp4', 'rb')
            st.audio(audio, format='audio/mp4')

            with st.spinner("Processing. If video is long, this may take a while - around 4 minutes for a 20 minute video."):

                transcript = generate_subtitles(stream, model, test=True)
                process_video = True

                st.success("Transcription is done. File saved as transcript.txt")

# main page

st.title(":video_camera: YouTube Video Q and A")

st.write("This is an app that will allow you to summarise youtube videos and chat with them")
st.write("We use whisper to get the transcript of the video and then use the openai api to summarise the video")
st.write("We then use the openai api and langchain to chat with the video")

st.write("<h2>How to use the app</h2>", unsafe_allow_html=True)
st.write("1. Enter your openai api key in the sidebar - this is only for summarisation and chatting with the video")
st.write("2. Enter the youtube url of the video from Youtube")
st.write("3. Click on the start analysis button")

st.header("Summary")

if transcript != "":

    llm = OpenAI(temperature=0, openai_api_key=user_secret)

    with st.spinner("Generating summary"):

        st.session_state.summary = generate_summary(transcript, llm)

    # print(summary)
    with open('summary.txt', 'w') as f:
        f.write(st.session_state.summary)

    st.success("Summary generated")
    st.write("This is the summary of the video")
    st.write(st.session_state.summary)
else:
    pass

process_question = None

# @st.cache_data
# def call_get_answer(question):
#     return get_answer(question)

if 'process_question_clicked' not in st.session_state:
    st.session_state.process_question_clicked = False

def process_question_callback():
    st.session_state.process_question_clicked = True


# if transcript != "":
#     col_1, col_2 = st.columns([0.8, 0.2])
#     with col_1:
#         question = st.text_input(label='Question', label_visibility='collapsed')
#     with col_2:
#         process_question = st.button('Get Answer', on_click=process_question_callback)


# if process_question or st.session_state.process_question_clicked:
# st.write(st.session_state.transcript)
if st.session_state.summary != "":
    # container_3 = st.container()
    # with container_3:
    #     with st.spinner('Finding answer...'):
    #         status, data = call_get_answer(question)
    #         if status != 'success':
    #             st.error(data)
    #             exit(0)
    #         else:
    #             answer = data
    #     st.text('The answer to your question: ')

    #     answer_box = st.text_area(
    #         label='Answer',
    #         label_visibility='collapsed',
    #         value=answer,
    #         disabled=True,
    #         height=300)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    documents = text_splitter.create_documents([transcript])

    # st.write(documents)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embedding_function)
    vectorstore.persist()

    st.write(vectorstore.get()['ids'])

    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.write(vectorstore.embeddings)

    # Accept user input
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # process_question_callback()
        with st.chat_message("user"):
            st.markdown(prompt)
        st.write(vectorstore.embeddings)

        # Query the assistant using the latest chat history
        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
        # st.write(qa({"question": "What OpenAI aims to evalute"}))
        # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     message_placeholder = st.empty()
        #     # full_response = ""
        #     full_response = result["answer"]
        #     message_placeholder.markdown(full_response + "|")
        # message_placeholder.markdown(full_response)
        # print(full_response)
        # st.session_state.messages.append({"role": "assistant", "content": full_response})

        # sleep 30 seconds
        time.sleep(30)
