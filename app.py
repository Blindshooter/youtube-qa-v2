import streamlit as st
import os

from youtube_utils import get_video_info, generate_subtitles
from functions import generate_summary  # , get_answer
import whisper

from langchain.llms import OpenAI

process_video = None
process_question = None
go_to_timestamp = None

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

        summary = generate_summary(transcript, llm)

    # print(summary)
    with open('summary.txt', 'w') as f:
        f.write(summary)

    st.success("Summary generated")
    st.write("This is the summary of the video")
    st.write(summary)
else:
    pass

process_question = None

@st.cache_data
def call_get_answer(question):
    return get_answer(question)

if 'process_question_clicked' not in st.session_state:
    st.session_state.process_question_clicked = False

def process_question_callback():
    st.session_state.process_question_clicked = True


if summary != "":
    col_1, col_2 = st.columns([0.8, 0.2])
    with col_1:
        question = st.text_input(label='Question', label_visibility='collapsed')
    with col_2:
        process_question = st.button('Get Answer', on_click=process_question_callback)


# program variables and functions

# @st.cache_data
# def call_get_summary(transcript, _llm):
#     return generate_summary(transcript, llm)


# @st.cache_data
# def call_get_answer(question):
#     return get_answer(question)

# # session state management

# if process_video or st.session_state.process_video_clicked:
#     if yt_url == "":
#         st.error("Please provide a valid link!")
#         exit(0)

#     st.session_state.process_video_clicked = True
#     st.divider()
#     container_1 = st.container()
#     with container_1:
#         with st.spinner('Generating summary...'):
#             llm = OpenAI(temperature=0, openai_api_key=user_secret)
#             summary = generate_summary(transcript, llm)
#         if summary == '':
#             st.error(summary)
#             exit(0)
#         else:
#             st.text('Summary of the video:')
#             summary_box = st.text_area(
#                 label='Summary',
#                 label_visibility='collapsed',
#                 value=summary,
#                 disabled=True,
#                 height=300)
#         st.divider()
#         st.text('Type your question here: ')
        # col_1, col_2 = st.columns([0.8, 0.2])
        # with col_1:
        #     question = st.text_input(
        #         label = 'Question',
        #         label_visibility = 'collapsed',
        #         )
        # with col_2:
        #     process_question = st.button('Get Answer', on_click=process_question_callback)

# answer container

if process_question or st.session_state.process_question_clicked:
    container_3 = st.container()
    with container_3:
        with st.spinner('Finding answer...'):
            status, data = call_get_answer(question)
            if status != 'success':
                st.error(data)
                exit(0)
            else:
                answer, st.session_state.timestamp = data[0], data[1]
        st.text('The answer to your question: ')

        answer_box = st.text_area(
            label='Answer',
            label_visibility='collapsed',
            value=answer,
            disabled=True,
            height=300)
