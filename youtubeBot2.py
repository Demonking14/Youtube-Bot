import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yt_dlp
import requests
import os
from dotenv import load_dotenv
import re
st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🤖")
load_dotenv()
def get_transcript(url):
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'quiet': True,
        'no_warnings': True,
        'force_ipv4': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("requested_subtitles")
            sub_url = None
            
            if subs and "en" in subs:
                sub_url = subs["en"]["url"]
            else:
             
                available = info.get("subtitles") or {}
                auto_caps = info.get("automatic_captions") or {}
                combined = {**auto_caps, **available}
                
                for lang_code in ["en", "en-US", "en-GB", "en-CA", "en-IN"]:
                    if lang_code in combined:
                        formats = combined[lang_code]
                        sub_url = next((f["url"] for f in formats if f.get("ext") == "vtt"), formats[0]["url"])
                        break
            
            if not sub_url:
                return None

            response = requests.get(sub_url)
            response.raise_for_status()
            data = response.text
            
         
            lines = []
            for line in data.split("\n"):
                line = line.strip()
             
                if (line.startswith("WEBVTT") or "-->" in line or line.isdigit() or not line):
                    continue
                
                line = re.sub(r'<[^>]*>', '', line)
                lines.append(line)
                
            return " ".join(lines)
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

@st.cache_resource
def build_vectorstore(transcript):
    if not transcript:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions from a YouTube video transcript.

Chat History:
{history}

Context:
{context}

Question: {question}
Answer only using the provided context. If the answer is not in the context, say you don't know.
""",
    input_variables=["history", "context", "question"],
)

st.title("📺 YouTube Video Chatbot")
st.markdown("Chat with any YouTube video that has English subtitles/captions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")


if "last_url" not in st.session_state or st.session_state.last_url != url:
    if "retriever" in st.session_state:
        del st.session_state.retriever
    st.session_state.last_url = url
    st.session_state.messages = []

if url:
    if "retriever" not in st.session_state:
        with st.spinner("🔍 Fetching transcript and indexing..."):
            transcript = get_transcript(url)
            if transcript:
                vector_store = build_vectorstore(transcript)
                if vector_store:
                    st.session_state.retriever = vector_store.as_retriever()
                    st.success("✅ Transcript indexed! You can now ask questions.")
                else:
                    st.error("❌ Failed to process transcript.")
            else:
                st.error("❌ Could not find English subtitles for this video.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_history():
    return "\n".join(
        [f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
         for m in st.session_state.messages]
    )

def ask_question(question):
    if "retriever" not in st.session_state:
        return "Please provide a valid YouTube URL first."
    
    retriever = st.session_state.retriever
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = prompt | llm | parser

    return chain.invoke({
        "history": get_history(),
        "context": context,
        "question": question
    })


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if user_input := st.chat_input("Ask about the video..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            response = ask_question(user_input)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
