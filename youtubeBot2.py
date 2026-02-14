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

# Page configuration
st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🤖")

# Load environment variables
load_dotenv()

# API Key handling: Priority to Streamlit Secrets, then environment variables
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.sidebar.warning("⚠️ GOOGLE_API_KEY not found in secrets or .env")
    api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

if not api_key:
    st.info("Please provide a Google API Key to continue.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

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
            
            if not subs or "en" not in subs:
                # Try fallback: some videos have 'en-US' or similar
                available_subs = info.get("subtitles") or info.get("automatic_captions")
                if available_subs:
                    for lang in ['en', 'en-US', 'en-GB']:
                        if lang in available_subs:
                            # This is a bit complex for a simple script, 
                            # sticking to basic 'en' for now or returning None
                            pass
                return None
            
            sub_url = subs["en"]["url"]
            response = requests.get(sub_url)
            response.raise_for_status()
            data = response.text
            
            # Simple VTT/SRT parsing to text
            lines = [line for line in data.split("\n") if "-->" not in line and line.strip() and not line.isdigit()]
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
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Prompt Template
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

# Clear retriever if URL changes
if "last_url" not in st.session_state or st.session_state.last_url != url:
    if "retriever" in st.session_state:
        del st.session_state.retriever
    st.session_state.last_url = url
    st.session_state.messages = [] # Reset chat for new video

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
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
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

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about the video..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            response = ask_question(user_input)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
