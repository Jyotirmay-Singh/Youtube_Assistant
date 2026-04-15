# Changes to make it run on streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.moudles.pop('pysqlite3')


import streamlit as st
from supporting_functions import extract_video_id, get_transcript, generate_notes, get_important_topics, translate_transcript, create_chunks, create_vector_store, rag_answer

# Dictionary to map full language names to language codes
LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "korean": "ko",
    "japanese": "ja",
    "chinese": "zh",
    "russian": "ru",
    "portuguese": "pt"
}

with st.sidebar:
    st.title("🎬 Youtube Video Assistant")
    st.markdown("---")
    st.markdown("Transform any Youtube video into key topics, podcast, or chatbot")
    st.markdown("### Input Details")

    youtube_url = st.text_input("Youtube URL", placeholder="https://www.youtube.com/watch?v=...")
    language_input = st.text_input("Video Language", placeholder="e.g., English, Hindi, Spanish", value="English")

    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes for you"]
    )

    submit_button = st.button("Start Processing: ")
    st.markdown("---")

# --- Main Page ----
st.title("✨ Youtube Content Synthesizer")
st.markdown("Paste a video link and select a task from the sidebar")

# --- Processing Flow ---

if submit_button:
    if youtube_url and language_input:
        
        # Clean input and default to "en" if the language is not in the dictionary
        cleaned_lang_input = language_input.strip().lower()
        lang_code = LANGUAGE_MAP.get(cleaned_lang_input, "en") 

        video_id = extract_video_id(youtube_url)
        
        if video_id:
            with st.spinner("Step 1/3: Fetching Transcript ......"):
                full_transcript = get_transcript(video_id, lang_code)
                
                if lang_code != "en":
                    with st.spinner("Step 1.5/3: Translating Transcript into English. This may take a few moments ......"):
                        full_transcript = translate_transcript(full_transcript)

            if task_option == "Notes for you": 
                with st.spinner("Step 2/3: Extracting important topics ..."):
                    important_topics = get_important_topics(full_transcript)
                    st.subheader("Important Topics")
                    st.write(important_topics)

                with st.spinner("Step 3/3: Generating Notes for you "):
                    notes = generate_notes(full_transcript)
                    st.subheader("Notes for you")
                    st.write(notes)
                
                st.success("Summary and Notes Generated")
                
            elif task_option == "Chat with Video":
                with st.spinner("Step 2/3: Creating Chunks and Vector Space ...."):
                    chunks = create_chunks(full_transcript)
                    vectorstore = create_vector_store(chunks)
                    st.session_state.vector_store = vectorstore
                st.session_state.messages=[]
                st.success('Video is ready for chat ...')


# chatbot session
if task_option == "Chat with Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chat with video")

    # display the entire history
    for message in st.session_state.get('messages',[]):
        with st.chat_message(message['role']):
            st.write(message['content'])



    # user input
    prompt = st.chat_input("Ask me anything about the video.")
    if prompt:
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = rag_answer(prompt, st.session_state.vector_store)
            st.write(response)
        st.session_state.messages.append({'role':'assistant','content':response})
