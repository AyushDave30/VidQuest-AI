import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from datetime import timedelta
import requests
import json
import requests
import xml.etree.ElementTree as ET
import os


# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv()  # works locally if .env is present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "❌ OPENAI_API_KEY not found! Please set it in .env (local) or Streamlit Secrets (cloud)."
    )
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------------------------
# 2. Streamlit Config & CSS
# -----------------------------
st.set_page_config(page_title="🎥 YouTube RAG Chatbot", page_icon="🎥", layout="wide")
st.markdown(
    """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    margin-bottom: 6px;
}
.user-msg {
    align-self: flex-end;
    background-color: #2b5278; /* Dark muted blue */
    color: #ffffff;
    padding: 8px 12px;
    border-radius: 12px 12px 0 12px;
    max-width: 70%;
    margin: 3px 0;
}
.bot-msg {
    align-self: flex-start;
    background-color: #3a3b3c; /* Dark gray */
    color: #e4e6eb;
    padding: 8px 12px;
    border-radius: 12px 12px 12px 0;
    max-width: 70%;
    margin: 3px 0;
}
.source-snippet {
    font-size:13px; 
    margin-left:10px; 
    color: #ccc;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎥 YouTube RAG Chatbot")
st.write("Chat with any YouTube video based on its transcript!")

# -----------------------------
# 3. Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "video_id" not in st.session_state:
    st.session_state["video_id"] = None
if "timed_chunks" not in st.session_state:
    st.session_state["timed_chunks"] = None
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False


# -----------------------------
# 4. Helper Functions
# -----------------------------


@st.cache_data(show_spinner="📥 Fetching transcript via ScraperAPI...")
def get_transcript(video_id: str):
    scraper_key = os.getenv("SCRAPER_API_KEY")

    # YouTube transcript endpoint (returns XML, not JSON)
    youtube_transcript_url = f"https://www.youtube.com/api/timedtext?lang=en&v={video_id}"
    proxy_url = f"http://api.scraperapi.com?api_key={scraper_key}&url={youtube_transcript_url}"

    response = requests.get(proxy_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch transcript: {response.status_code}")

    # ✅ Parse XML instead of JSON
    root = ET.fromstring(response.text)

    timed_chunks = []
    all_text = []

    for child in root.findall("text"):
        text = (child.text or "").replace("\n", " ").strip()
        start = float(child.attrib.get("start", 0))
        if text:
            timed_chunks.append({"text": text, "start": start})
            all_text.append(text)

    if not timed_chunks:
        raise Exception("Transcript not available for this video.")

    return " ".join(all_text), timed_chunks


@st.cache_resource(show_spinner="🔗 Building vector store...")
def build_vector_store_from_text(_transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([_transcript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def seconds_to_hms(seconds: float):
    return str(timedelta(seconds=int(seconds)))


def answer_question(question: str, retriever, llm):
    retrived_docs = retriever.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    ).invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    prompt = f"""You are a helpful assistant.
Answer ONLY based on the transcript context below.
If the answer is not in the context, say you don't know.

Context:
{context_text}

Question: {question}"""
    answer = llm.invoke(prompt)
    return answer.content, retrived_docs


def map_chunks_to_timestamps(retrived_docs, timed_chunks):
    mapped = []
    for doc in retrived_docs:
        for t in timed_chunks:
            if t["text"] in doc.page_content:
                mapped.append((t["text"], seconds_to_hms(t["start"])))
    mapped = list({(txt, ts) for txt, ts in mapped})
    mapped = sorted(mapped, key=lambda x: x[1])
    return mapped[:3]


def summarize_video(retriever, llm):
    all_docs = retriever.as_retriever(
        search_type="similarity", search_kwargs={"k": 20}
    ).invoke("main points of this video")
    context_text = "\n\n".join(doc.page_content for doc in all_docs)
    prompt = f"""Summarize the transcript into the TOP 5 key points in concise bullet points.
Context:
{context_text}"""
    summary = llm.invoke(prompt)
    return summary.content


# Process user question
def process_question():
    user_question = st.session_state["user_question"].strip()
    if user_question:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer, retrived_docs = answer_question(
            user_question, st.session_state["retriever"], llm
        )
        st.session_state["messages"].append(
            {"question": user_question, "answer": answer, "sources": retrived_docs}
        )
        st.session_state["user_question"] = ""  # ✅ Safe since it's before rendering


# -----------------------------
# 5. Video URL Input
# -----------------------------
video_url = st.text_input("Paste YouTube Video URL:")

if video_url:
    if st.session_state.get("video_id") != video_url:
        try:
            # 🔥 Force clear previous session data when URL changes
            st.cache_data.clear()
            st.cache_resource.clear()

            video_id = video_url.split("v=")[-1].split("&")[0]
            transcript, timed_chunks = get_transcript(video_id)
            vector_store = build_vector_store_from_text(transcript)

            st.session_state["retriever"] = vector_store
            st.session_state["video_id"] = video_url
            st.session_state["timed_chunks"] = timed_chunks
            st.session_state["messages"].clear()

            st.success("✅ Transcript processed! You can start asking questions.")
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ✅ Clear input before rendering
if st.session_state.get("clear_input"):
    st.session_state["user_question"] = ""
    st.session_state["clear_input"] = False

# -----------------------------
# 6. Chat UI
# -----------------------------
st.text_input(
    "Ask a Question about the video:",
    key="user_question",
    placeholder="Type your question here and press Enter...",
    on_change=process_question,
)

if st.button("🔝 Summarize Video (Top 5 Points)", key="summary_btn"):
    st.session_state["clear_input"] = True
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    summary = summarize_video(st.session_state["retriever"], llm)
    st.session_state["messages"].append(
        {
            "question": "🔝 Summary of Video (Top 5 Points)",
            "answer": summary,
            "sources": [],
        }
    )
    st.rerun()

if st.session_state["messages"]:
    for i, msg in enumerate(st.session_state["messages"]):
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='user-msg'>{msg['question']}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='bot-msg'>{msg['answer']}</div>", unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if msg["sources"]:
            if st.button("📄 Sources", key=f"sources_{i}"):
                sources = map_chunks_to_timestamps(
                    msg["sources"], st.session_state["timed_chunks"]
                )
                for txt, ts in sources:
                    snippet = (txt[:100] + "...") if len(txt) > 100 else txt
                    st.markdown(
                        f"<div class='source-snippet'>✅ **[{ts}]** {snippet}</div>",
                        unsafe_allow_html=True,
                    )

if st.button("🗑️ Clear Chat History"):
    st.session_state["messages"].clear()
    st.rerun()
