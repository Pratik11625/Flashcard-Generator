import os
import io
import sys
import json
import streamlit as st
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
# import matplotlib.pyplot as plt
# from langchain_ollama import Ollama
import requests
import time
import pandas as pd



load_dotenv()

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Flashcard Generator", page_icon="📚")

# timer
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "time_limit" not in st.session_state:
    st.session_state.time_limit = 15  # seconds per question


# --------------------------------------------------
# Sidebar: API Key
# # --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    use_env_key = st.toggle("Use System API Key")



    if use_env_key:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY:
            st.success("✅ API Key loaded from environment")
        else:
            st.error("❌ GROQ_API_KEY not found in .env")
            GROQ_API_KEY = None
    else:
        GROQ_API_KEY = st.text_input("Enter Groq API Key", type="password")

# --------------------------------------------------
# Load Data helpers
# --------------------------------------------------
DATA_FILE = "data.json"

@st.cache_data
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_uploaded_data(uploaded_file):
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    return json.loads(raw.decode("utf-8"))

def validate_data(data):
    required = {"unit", "topic", "subtopic", "text"}
    errors = []
    for i, entry in enumerate(data):
        missing = required - set(entry.keys())
        if missing:
            errors.append(f"Entry {i+1} missing fields: {missing}")
    return errors

# --------------------------------------------------
# Sidebar: File Upload
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload JSON")
st.sidebar.markdown("""
Upload a JSON file structured as:
```json
[
  {
    "unit":     "Unit 1",
    "topic":    "AI",
    "subtopic": "Definition",
    "text":     "Your content..."
  }
]
```
""")

uploaded_file = st.sidebar.file_uploader("Upload data.json", type=["json"])

# --------------------------------------------------
# Load Data
# --------------------------------------------------
if uploaded_file:
    try:
        data = load_uploaded_data(uploaded_file)
        errs = validate_data(data)
        if errs:
            st.sidebar.error("❌ Errors:\n" + "\n".join(errs))
            st.stop()
        st.sidebar.success(f"✅ Loaded {len(data)} entries")
    except Exception as e:
        st.sidebar.error(f"❌ Could not read file: {e}")
        st.stop()
elif os.path.exists(DATA_FILE):
    try:
        data = load_data(DATA_FILE)
    except Exception as e:
        st.error(f"❌ Could not read data.json: {e}")
        st.stop()
else:
    st.error("⚠️ No data.json found. Place it in the same folder as app.py or upload via sidebar.")
    st.stop()

# --------------------------------------------------
# Vector DB
# --------------------------------------------------
@st.cache_resource
def build_vector_store(data_json):
    data = json.loads(data_json)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = Client(Settings(anonymized_telemetry=False))

    try:
        client.delete_collection("flashcards")
    except Exception:
        pass

    collection = client.create_collection("flashcards")

    texts      = [d["text"] for d in data]
    ids        = [str(i) for i in range(len(data))]
    metadatas  = [{"unit": d["unit"], "topic": d["topic"], "subtopic": d["subtopic"]} for d in data]
    embeddings = model.encode(texts).tolist()

    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection, model

with st.spinner("Building vector DB..."):
    collection, embed_model = build_vector_store(json.dumps(data))

# --------------------------------------------------
# Retrieve
# --------------------------------------------------
def retrieve_content(collection, model, query, unit, topic, subtopic):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        where={"$and": [{"unit": unit}, {"topic": topic}, {"subtopic": subtopic}]}
    )
    docs = results.get("documents", [[]])[0]
    return " ".join(docs)




# --------------------------------------------------
# LLM (Groq)
# --------------------------------------------------
def generate_flashcards(content, api_key):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=api_key)
    messages = [
        SystemMessage(content="You are a helpful teacher."),
        HumanMessage(content=f"""
Create {number} high-quality flashcards.

Rules:
- Mix conceptual and factual questions
- Keep answers concise (1-2 sentences max)
- No repetition
- Format EXACTLY as:
Q: question
A: answer

Content:
{content}
""")
    ]
    response = llm.invoke(messages)
    return response.content

# --------------------------------------------------
# Parse
# --------------------------------------------------
def parse_flashcards(text):
    cards = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    current = {}
    for line in lines:
        if line.startswith("Q:"):
            if current.get("q") and current.get("a"):
                cards.append(current)
            current = {"q": line[2:].strip(), "a": ""}
        elif line.startswith("A:"):
            current["a"] = line[2:].strip()
    if current.get("q") and current.get("a"):
        cards.append(current)
    return cards


# timer
def next_card(action):
    status[idx] = action
    if idx < len(cards)-1:
        st.session_state.idx += 1
        st.session_state.flip = False
        st.session_state.start_time = time.time()  # 🔥 reset timer

def power():
    if "status" in st.session_state and st.session_state.status:
        status = st.session_state.status

        known = list(status.values()).count("known")
        revision = list(status.values()).count("revision")
        skip = list(status.values()).count("skip")

        total = len(st.session_state.cards) if "cards" in st.session_state else 0

        return 0 + (10 * known)

    return 0

# --------------------------------------------------
# Main UI
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
xp = power()
c3.metric("Your XP", f"{xp} XP")


st.title("📚 Flashcard Generator")

col1, col2, col3 = st.columns(3)

units     = sorted(set(d["unit"] for d in data))
unit      = col1.selectbox("Unit", units)

topics    = sorted(set(d["topic"] for d in data if d["unit"] == unit))
topic     = col2.selectbox("Topic", topics)

subtopics = sorted(set(d["subtopic"] for d in data if d["unit"] == unit and d["topic"] == topic))
subtopic  = col3.selectbox("Subtopic", subtopics)

number = st.slider("Number of Flashcards", min_value=5, max_value=20, value=10, step=1)  # New slider for number of flashcards
number = int(number)  # Ensure it's an integer for the prompt

st.markdown("---")

# --------------------------------------------------
# Generate
# --------------------------------------------------
if st.button("⚡ Generate Flashcards", use_container_width=True):
    if not GROQ_API_KEY:
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Generating flashcards..."):
            query   = f"{topic} {subtopic}"
            content = retrieve_content(collection, embed_model, query, unit, topic, subtopic)
            if not content.strip():
                st.error("No matching content found for this selection.")
            else:
                try:
                    raw   = generate_flashcards(content, GROQ_API_KEY)
                    cards = parse_flashcards(raw)
                    if not cards:
                        st.error("Could not parse flashcards. Try again.")
                    else:
                        st.session_state.cards  = cards
                        st.session_state.idx    = 0
                        st.session_state.flip   = False
                        st.session_state.status = {}
                except Exception as e:
                    st.error(f"LLM error: {e}")

# --------------------------------------------------
# Display Flashcards
# --------------------------------------------------
if "cards" in st.session_state and st.session_state.cards:
    cards  = st.session_state.cards
    idx    = st.session_state.idx
    flip   = st.session_state.flip
    status = st.session_state.status
    card   = cards[idx]

    st.markdown("---")
    st.subheader(f"Card {idx+1} / {len(cards)}")
    st.progress((idx + 1) / len(cards))


    with st.container(border=True):
        if not flip:
            c1, c2 = st.columns([1, 4])


            if len(card["q"]) > 0:
                # c1.markdown("**Question**")
            # c2.markdown(f"### {idx + 1} / {len(cards)}")
                c1.markdown(f"### Question {idx + 1} of {len(cards)}")
            

                st.markdown(f"### {card['q']}")
            # st.markdown(f"### {card['q']}")

            if st.button("🔄 Reveal Answer", use_container_width=True):
                st.session_state.flip = True
                st.rerun()
        else:
            st.markdown("**Answer**")
            st.markdown(f"### {card['a']}")
            if st.button("🔄 Show Question", use_container_width=True):
                st.session_state.flip = False
                st.rerun()

    st.markdown("**How well did you know this?**")
    c1, c2, c3 = st.columns(3)

    def next_card(action):

        st.session_state.status[idx] = action
        if idx < len(cards) - 1:
            st.session_state.idx  += 1
            st.session_state.flip  = False

    if c1.button("✅ I Know", use_container_width=True):
        next_card("known");    st.rerun()

    if c2.button("🔁 Revision", use_container_width=True):
        next_card("revision"); st.rerun()

    if c3.button("⏭ Skip", use_container_width=True):
        next_card("skip");     st.rerun()

    n1, n2 = st.columns(2)
    if n1.button("← Previous", use_container_width=True, disabled=(idx == 0)):
        st.session_state.idx  -= 1
        st.session_state.flip  = False
        st.rerun()
    
    if n2.button("Next →", use_container_width=True, disabled=(idx == len(cards) - 1)):
        st.session_state.idx  += 1
        st.session_state.flip  = False
        st.rerun()


    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False 

    if st.button("Submit", use_container_width=True):

        end_time = time.time()
        time_taken = end_time - st.session_state.start_time 

        # st.stop()  # Stop interactions to show results
        st.session_state.quiz_completed = True  # 🔥 lock system
        st.success(f"⏱ Time taken: {time_taken:.2f} seconds")
        
        st.markdown("--Thank you--")

      
        if xp > 100:
            st.success("🔥 Pro Learner")
        elif xp > 50:
            st.info("🚀 Improving Fast")
        else:
            st.warning("📚 Keep Practicing")

       

    
# --------------------------------------------------
# Dashboard
# --------------------------------------------------
if st.session_state.get("status"):
    status = st.session_state.status
    total  = len(st.session_state.cards)
    known    = list(status.values()).count("known")
    revision = list(status.values()).count("revision")
    skip     = list(status.values()).count("skip")

    st.markdown("---")
    st.subheader("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    # c1.metric("✅ Known",    f"{known}/{total}")
    # c2.metric("🔁 Revision", f"{revision}/{total}")
    # c3.metric("⏭ Skipped",  f"{skip}/{total}")
    
    c1.metric("✅ Known",    f"{known}")
    c2.metric("🔁 Revision", f"{revision}")
    c3.metric("⏭ Skipped",  f"{skip}")

    c1.metric("Known %", f"{(known/total)*100:.0f}%")
    c2.metric("Revision %", f"{(revision/total)*100:.0f}%")
    c3.metric("Skipped %", f"{(skip/total)*100:.0f}%")

    # st.success('you have scored')
    st.markdown("---")
    with st.container(border=True):
        c1, c2 ,c3 = st.columns(3)
        
        c1.metric("Score", f"{((known*2 + revision) / (total*2))*100:.0f}%")
        c2.metric("Skipped %", f"{(skip/total)*100:.0f}%")
        c3.metric("Total Cards", f"{known + revision + skip}/{total}")


    score = ((known*2 + revision) / (total*2)) * 100

    if score > 80:
        st.success("Excellent Performance 🚀")
    elif score > 50:
        st.info("Good, but needs improvement")
    else:
        if revision > known:
            st.warning("You need more revision in this topic!")
        st.error("Focus more on revision")


    if st.button("🔄 Reset Progress", use_container_width=True):
        st.session_state.status = {}
        st.session_state.idx    = 0
        st.session_state.flip   = False
        st.rerun()
    elif st.button("📁 New Selection", use_container_width=True):
        del st.session_state.cards
        st.session_state.idx    = 0
        st.session_state.flip   = False
        st.session_state.status = {}
        st.rerun()
    else:
        st.info("Use the buttons above to reset or choose a new topic.")
