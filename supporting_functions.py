import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# initialize the gemini model
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.4
)


def extract_video_id(url):
    """
    Extracts the youtube video ID from any valid Youtube URL.
    """

    match = re.search(pattern= r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",string=url)
    if match:
        return match.group(1)
    st.error("Invalid Youtube URL. Please enter a valid video link.")
    return None

def get_transcript(video_id,language="en"):
    ytt_api = YouTubeTranscriptApi()

    try:
        transcript=ytt_api.fetch(video_id,languages=[language])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(10)
        return full_transcript
    except Exception as e:
        st.error(f" Error fetching video {e}")

# Function to translate the transcript into english
def translate_transcript(transcript):
    try:
        prompt=ChatPromptTemplate.from_template("""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker’s voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """)

        # Runnable chain
        chain=prompt|llm

        # Run chain
        response=chain.invoke({"transcript":transcript})
        return response.content
    
    except Exception as e:
        st.error(f"Error fetching video {e}")


# function to get important topics
def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """)
        
        # Runnable chain
        chain = prompt | llm

        # Run chain
        response = chain.invoke({"transcript":transcript})
        return response.content
    
    except Exception as e:
        st.error(f"Error encountered while topics acquiring: {e}")

def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
                You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                ⚡ Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}
                """)

        # Runnable chain
        chain = prompt | llm

        # Run chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error fething video {e}")


# Function to create chunks
def create_chunks(transcript):
    text_splitters=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    doc = text_splitters.create_documents([transcript])
    return doc

def create_vector_store(docs):
    embedding= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", transport="grpc" )
    vector_store = Chroma.from_documents(docs,embedding)
    return vector_store

# RAG Function

def rag_answer(question,vectorStore):
    results = vectorStore.similarity_search(question,k=3)
    context_text = "\n".join([i.page_content for i in results])

    prompt = ChatPromptTemplate.from_template("""
                You are a kind, polite, and precise assistant.
                - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
                - Understand the user’s intent even with typos or grammatical mistakes.
                - Answer ONLY using the retrieved context.
                - If answer not in context, say:
                  "I couldn’t find that information in the database. Could you please rephrase or ask something else?"
                - Keep answers clear, concise, and friendly.

                Context:
                {context}

                User Question:
                {question}

                Answer:
                """)

    #chain
    chain = prompt|llm
    response= chain.invoke({"context":context_text,"question":question})

    return response.content