import os
import streamlit as st
from dotenv import load_dotenv
import subprocess
import json
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from datetime import timedelta
import re
from urllib.parse import urlparse, parse_qs

# -----------------------------
# 1. Load API Key
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "❌ OPENAI_API_KEY not found! Please set it in .env (local) or Streamlit Secrets (cloud)."
    )
    st.stop()
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
    background-color: #2b5278;
    color: #ffffff;
    padding: 8px 12px;
    border-radius: 12px 12px 0 12px;
    max-width: 70%;
    margin: 3px 0;
}
.bot-msg {
    align-self: flex-start;
    background-color: #3a3b3c;
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

st.title("🎥 YouTube RAG Chatbot (Alternative)")
st.write("Chat with any YouTube video based on its transcript using yt-dlp!")

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

# -----------------------------
# 4. Helper Functions
# -----------------------------


def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
        r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
        elif parsed_url.hostname == "youtu.be":
            return parsed_url.path[1:]
    except:
        pass

    return None


@st.cache_data(show_spinner="📥 Fetching transcript using yt-dlp...")
def get_transcript_ytdlp(video_id: str):
    """Get transcript using yt-dlp - more reliable alternative"""
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Create temporary directory for yt-dlp output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use yt-dlp to extract subtitle information
            cmd = [
                "yt-dlp",
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs",
                "en",
                "--skip-download",
                "--output",
                f"{temp_dir}/%(title)s.%(ext)s",
                video_url,
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                # Look for subtitle files
                import glob

                vtt_files = glob.glob(f"{temp_dir}/*.vtt")

                if not vtt_files:
                    raise Exception("No subtitle files found")

                # Parse VTT file
                subtitle_file = vtt_files[0]
                timed_chunks = []
                all_text = []

                with open(subtitle_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse VTT format
                lines = content.split("\n")
                current_text = ""
                current_start = 0

                for line in lines:
                    line = line.strip()
                    if "-->" in line:
                        # Time line
                        time_parts = line.split(" --> ")
                        start_time = time_parts[0]
                        # Convert time to seconds
                        time_obj = start_time.split(":")
                        if len(time_obj) >= 3:
                            current_start = (
                                float(time_obj[0]) * 3600
                                + float(time_obj[1]) * 60
                                + float(time_obj[2])
                            )
                    elif line and not line.startswith("WEBVTT") and not line.isdigit():
                        # Text line
                        clean_text = re.sub(r"<[^>]+>", "", line)  # Remove HTML tags
                        if clean_text:
                            timed_chunks.append(
                                {"text": clean_text, "start": current_start}
                            )
                            all_text.append(clean_text)

                if not timed_chunks:
                    raise Exception("No valid transcript content found")

                return " ".join(all_text), timed_chunks

            except subprocess.TimeoutExpired:
                raise Exception(
                    "yt-dlp timeout - video might be too long or unavailable"
                )
            except FileNotFoundError:
                raise Exception(
                    "yt-dlp not found. Please install it: pip install yt-dlp"
                )

    except Exception as e:
        raise Exception(f"Failed to get transcript: {str(e)}")


@st.cache_data(show_spinner="📥 Fetching transcript (fallback method)...")
def get_transcript_simple(video_id: str):
    """Simple fallback using youtube-transcript-api with basic error handling"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Get transcript data
        transcript_data = YouTubeTranscriptApi().fetch(video_id)

        timed_chunks = []
        all_text = []

        for entry in transcript_data:
            # Handle both dict and object formats
            if isinstance(entry, dict):
                text = entry.get("text", "")
                start = entry.get("start", 0)
            else:
                text = getattr(entry, "text", str(entry))
                start = getattr(entry, "start", 0)

            text = text.replace("\n", " ").strip()
            if text:
                timed_chunks.append({"text": text, "start": float(start)})
                all_text.append(text)

        return " ".join(all_text), timed_chunks

    except Exception as e:
        raise Exception(f"Simple method failed: {str(e)}")


@st.cache_resource(show_spinner="🔗 Building vector store...")
def build_vector_store_from_text(_transcript: str):
    """Build FAISS vector store from transcript text"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        chunks = splitter.create_documents([_transcript])

        if not chunks:
            raise Exception("Failed to create text chunks")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise Exception(f"Failed to build vector store: {str(e)}")


def seconds_to_hms(seconds: float):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))


def answer_question(question: str, retriever, llm):
    """Answer question using RAG approach"""
    try:
        retrieved_docs = retriever.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ).invoke(question)

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        prompt = f"""You are a helpful assistant that answers questions based on YouTube video transcripts.

Answer the question based ONLY on the context provided below. If the answer is not clearly available in the context, say "I don't have enough information from the video transcript to answer that question."

Context from video transcript:
{context_text}

Question: {question}

Answer:"""

        answer = llm.invoke(prompt)
        return answer.content, retrieved_docs
    except Exception as e:
        return f"Error processing question: {str(e)}", []


def map_chunks_to_timestamps(retrieved_docs, timed_chunks):
    """Map retrieved chunks to their timestamps"""
    mapped = []
    for doc in retrieved_docs:
        doc_text = doc.page_content
        for chunk in timed_chunks:
            chunk_text = chunk["text"]
            if chunk_text in doc_text or doc_text in chunk_text:
                timestamp = seconds_to_hms(chunk["start"])
                mapped.append((chunk_text, timestamp))
                break

    mapped = list(dict.fromkeys(mapped))
    return mapped[:3]


def summarize_video(retriever, llm):
    """Generate video summary"""
    try:
        all_docs = retriever.as_retriever(
            search_type="similarity", search_kwargs={"k": 15}
        ).invoke("summarize main points key topics themes")

        context_text = "\n\n".join(doc.page_content for doc in all_docs)

        prompt = f"""Based on this YouTube video transcript, create a concise summary with the top 5 key points.

Transcript context:
{context_text}

Please provide:
1. A brief overview of what the video is about
2. The top 5 key points or main topics discussed
3. Keep each point concise and informative

Summary:"""

        summary = llm.invoke(prompt)
        return summary.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def process_question():
    """Process user question and add to chat history"""
    user_question = st.session_state.get("user_question", "").strip()
    if user_question and st.session_state["retriever"]:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            answer, retrieved_docs = answer_question(
                user_question, st.session_state["retriever"], llm
            )
            st.session_state["messages"].append(
                {"question": user_question, "answer": answer, "sources": retrieved_docs}
            )
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
        finally:
            st.session_state["user_question"] = ""


# -----------------------------
# 5. Main App Logic
# -----------------------------

# Method selection
method = st.selectbox(
    "Choose transcript extraction method:",
    ["Simple API", "yt-dlp (more reliable)"],
    help="If Simple API fails, try yt-dlp method",
)

# Video URL Input
video_url = st.text_input(
    "Paste YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=..."
)

if video_url:
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("❌ Invalid YouTube URL. Please enter a valid YouTube video URL.")
    elif st.session_state.get("video_id") != video_id:
        try:
            # Clear cache when processing new video
            st.cache_data.clear()
            st.cache_resource.clear()

            with st.spinner("Processing video transcript..."):
                if method == "yt-dlp (more reliable)":
                    transcript, timed_chunks = get_transcript_ytdlp(video_id)
                else:
                    transcript, timed_chunks = get_transcript_simple(video_id)

                vector_store = build_vector_store_from_text(transcript)

            # Update session state
            st.session_state["retriever"] = vector_store
            st.session_state["video_id"] = video_id
            st.session_state["timed_chunks"] = timed_chunks
            st.session_state["messages"] = []

            st.success(
                "✅ Video transcript processed successfully! You can now ask questions."
            )

            # Show video info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(video_url)
            with col2:
                st.metric("Transcript Length", f"{len(transcript.split())} words")
                st.metric("Time Segments", len(timed_chunks))
                st.metric("Method Used", method)

        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            if method == "Simple API":
                st.info("💡 Try switching to 'yt-dlp (more reliable)' method above")
            else:
                st.info("💡 Make sure yt-dlp is installed: pip install yt-dlp")

# Chat Interface (only show if video is loaded)
if st.session_state["retriever"]:
    st.markdown("---")

    # Question input
    st.text_input(
        "Ask a question about the video:",
        key="user_question",
        placeholder="What is this video about?",
        on_change=process_question,
    )

    # Summary button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📋 Summarize Video"):
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                summary = summarize_video(st.session_state["retriever"], llm)
                st.session_state["messages"].append(
                    {"question": "📋 Video Summary", "answer": summary, "sources": []}
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()

    # Display chat messages
    if st.session_state["messages"]:
        st.markdown("### 💬 Chat History")

        for i, msg in enumerate(st.session_state["messages"]):
            with st.container():
                st.markdown(f"**You:** {msg['question']}")
                st.markdown(f"**Assistant:** {msg['answer']}")

                # Show sources if available
                if msg["sources"] and st.session_state["timed_chunks"]:
                    with st.expander(f"📄 View Sources ({len(msg['sources'])} chunks)"):
                        sources = map_chunks_to_timestamps(
                            msg["sources"], st.session_state["timed_chunks"]
                        )
                        for j, (text, timestamp) in enumerate(sources):
                            snippet = text[:200] + "..." if len(text) > 200 else text
                            st.markdown(f"**[{timestamp}]** {snippet}")

                st.markdown("---")
else:
    st.info("👆 Enter a YouTube video URL above to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
🔧 **Methods:**
- **Simple API**: Uses youtube-transcript-api directly (faster but sometimes fails)
- **yt-dlp**: Downloads subtitle files directly from YouTube (more reliable)
)
