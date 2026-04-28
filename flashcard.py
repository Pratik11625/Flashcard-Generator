import os
import io
import json
import time
import streamlit as st
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flashcard Generator", page_icon="📚", layout="centered")
Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# ── Session State Defaults ────────────────────────────────────────────────────
for key, default in {
    "start_time"     : time.time(),
    "cards"          : [],
    "idx"            : 0,
    "flip"           : False,
    "status"         : {},
    "quiz_completed" : False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔐 Groq API Key")
    GROQ_API_KEY = st.text_input("API Key", type="password", placeholder="gsk_...")
    if GROQ_API_KEY:
        st.success("API Key added ✅")
    else:
        st.warning("Enter your Groq API key to continue")

    st.divider()
    st.subheader("📂 Upload JSON Data")
    st.markdown("""
Expected format:
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
    uploaded_file = st.file_uploader("Upload data.json", type=["json"])


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_uploaded_data(f) -> list:
    f.seek(0)
    raw = f.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    return json.loads(raw.decode("utf-8"))

def validate_data(data: list) -> list[str]:
    required = {"unit", "topic", "subtopic", "text"}
    return [
        f"Entry {i+1} missing: {required - set(entry.keys())}"
        for i, entry in enumerate(data)
        if required - set(entry.keys())
    ]

if not uploaded_file:
    st.info("👈 Upload a JSON file in the sidebar to get started.")
    st.stop()

try:
    data = load_uploaded_data(uploaded_file)
    errors = validate_data(data)
    if errors:
        st.sidebar.error("❌ Validation errors:\n" + "\n".join(errors))
        st.stop()
    st.sidebar.success(f"✅ {len(data)} entries loaded")
except Exception as e:
    st.sidebar.error(f"❌ Could not read file: {e}")
    st.stop()


# ── Vector Store ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base…")
def build_vector_store(data_json: str):
    data       = json.loads(data_json)
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    db         = Client(Settings(anonymized_telemetry=False))

    try:
        db.delete_collection("flashcards")
    except Exception:
        pass

    collection = db.create_collection("flashcards")
    texts      = [d["text"] for d in data]
    ids        = [str(i) for i in range(len(data))]
    metadatas  = [{"unit": d["unit"], "topic": d["topic"], "subtopic": d["subtopic"]} for d in data]
    embeddings = model.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection, model

collection, embed_model = build_vector_store(json.dumps(data))


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_content(query: str, unit: str, topic: str, subtopic: str) -> str:
    embedding = embed_model.encode([query]).tolist()
    results   = collection.query(
        query_embeddings=embedding,
        n_results=3,
        where={"$and": [{"unit": unit}, {"topic": topic}, {"subtopic": subtopic}]},
    )
    return " ".join(results.get("documents", [[]])[0])


# ── LLM ───────────────────────────────────────────────────────────────────────
def generate_flashcards(content: str, n: int, api_key: str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=api_key)
    messages = [
        SystemMessage(content="You are a helpful teacher."),
        HumanMessage(content=f"""
Create {n} high-quality flashcards.

Rules:
- Mix conceptual and factual questions
- Keep answers concise (1-2 sentences max)
- No repetition
- Format EXACTLY as:
Q: question
A: answer

Content:
{content}
"""),
    ]
    return llm.invoke(messages).content


def parse_flashcards(text: str) -> list[dict]:
    cards, current = [], {}
    for line in (l.strip() for l in text.split("\n") if l.strip()):
        if line.startswith("Q:"):
            if current.get("q") and current.get("a"):
                cards.append(current)
            current = {"q": line[2:].strip(), "a": ""}
        elif line.startswith("A:"):
            current["a"] = line[2:].strip()
    if current.get("q") and current.get("a"):
        cards.append(current)
    return cards


# ── XP helper ─────────────────────────────────────────────────────────────────
def calc_xp() -> int:
    return list(st.session_state.status.values()).count("known") * 10


# ── Main UI ───────────────────────────────────────────────────────────────────
xp_col, _, title_col = st.columns([1, 2, 1])
xp_col.metric("⚡ XP", f"{calc_xp()}")
st.title("📚 Flashcard Generator")

col1, col2, col3 = st.columns(3)
units     = sorted({d["unit"] for d in data})
unit      = col1.selectbox("Unit", units)
topics    = sorted({d["topic"] for d in data if d["unit"] == unit})
topic     = col2.selectbox("Topic", topics)
subtopics = sorted({d["subtopic"] for d in data if d["unit"] == unit and d["topic"] == topic})
subtopic  = col3.selectbox("Subtopic", subtopics)

number = st.slider("Number of Flashcards", 5, 20, 10)
st.markdown("---")

if st.button("⚡ Generate Flashcards", use_container_width=True, type="primary"):
    if not GROQ_API_KEY:
        st.warning("Enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Generating flashcards…"):
            content = retrieve_content(f"{topic} {subtopic}", unit, topic, subtopic)
            if not content.strip():
                st.error("No matching content found for this selection.")
            else:
                try:
                    raw   = generate_flashcards(content, number, GROQ_API_KEY)
                    cards = parse_flashcards(raw)
                    if not cards:
                        st.error("Could not parse flashcards. Try again.")
                    else:
                        st.session_state.cards          = cards
                        st.session_state.idx            = 0
                        st.session_state.flip           = False
                        st.session_state.status         = {}
                        st.session_state.quiz_completed = False
                        st.session_state.start_time     = time.time()
                        st.rerun()
                except Exception as e:
                    st.error(f"LLM error: {e}")


# ── Flashcard Display ─────────────────────────────────────────────────────────
if st.session_state.cards and not st.session_state.quiz_completed:
    cards  = st.session_state.cards
    idx    = st.session_state.idx
    status = st.session_state.status
    card   = cards[idx]

    st.markdown("---")
    st.progress((idx + 1) / len(cards), text=f"Card {idx+1} / {len(cards)}")

    # Card face
    with st.container(border=True):
        if not st.session_state.flip:
            st.markdown(f"#### ❓ Question {idx+1}")
            st.markdown(f"### {card['q']}")
            if st.button("🔄 Reveal Answer", use_container_width=True):
                st.session_state.flip = True
                st.rerun()
        else:
            st.markdown("#### ✅ Answer")
            st.markdown(f"### {card['a']}")
            if st.button("🔄 Show Question", use_container_width=True):
                st.session_state.flip = False
                st.rerun()

    # Rating buttons
    st.markdown("**How well did you know this?**")
    r1, r2, r3 = st.columns(3)

    def next_card(action: str):
        st.session_state.status[idx] = action
        if idx < len(cards) - 1:
            st.session_state.idx  += 1
            st.session_state.flip  = False
        st.rerun()

    if r1.button("✅ I Know",   use_container_width=True): next_card("known")
    if r2.button("🔁 Revision", use_container_width=True): next_card("revision")
    if r3.button("⏭ Skip",     use_container_width=True): next_card("skip")

    # Navigation
    n1, n2 = st.columns(2)
    if n1.button("← Previous", use_container_width=True, disabled=(idx == 0)):
        st.session_state.idx  -= 1
        st.session_state.flip  = False
        st.rerun()
    if n2.button("Next →", use_container_width=True, disabled=(idx == len(cards) - 1)):
        st.session_state.idx  += 1
        st.session_state.flip  = False
        st.rerun()

    st.markdown("---")
    if st.button("📊 Submit & See Results", use_container_width=True, type="primary"):
        st.session_state.quiz_completed = True
        st.session_state.time_taken     = time.time() - st.session_state.start_time
        st.rerun()


# ── Results Dashboard ─────────────────────────────────────────────────────────
if st.session_state.quiz_completed and st.session_state.cards:
    status   = st.session_state.status
    cards    = st.session_state.cards
    total    = len(cards)
    known    = list(status.values()).count("known")
    revision = list(status.values()).count("revision")
    skip     = list(status.values()).count("skip")
    score    = ((known * 2 + revision) / (total * 2)) * 100 if total else 0
    xp       = calc_xp()

    st.markdown("---")
    st.subheader("📊 Results")

    time_taken = st.session_state.get("time_taken", 0)
    st.info(f"⏱ Time taken: {time_taken:.1f}s")

    # Score summary
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Known",    known)
        c2.metric("🔁 Revision", revision)
        c3.metric("⏭ Skipped",  skip)
        c4.metric("⚡ XP",       f"{xp}")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Score",        f"{score:.0f}%")
        c2.metric("Known %",      f"{(known/total)*100:.0f}%" if total else "0%")
        c3.metric("Cards Done",   f"{known + revision + skip}/{total}")

    # Feedback
    if score > 80:
        st.success("🚀 Excellent Performance!")
    elif score > 50:
        st.info("👍 Good, keep going!")
    else:
        st.error("📚 Focus more on revision.")

    if xp > 100:
        st.success("🔥 Pro Learner")
    elif xp > 50:
        st.info("🚀 Improving Fast")
    else:
        st.warning("📚 Keep Practicing")

    st.markdown("---")
    b1, b2 = st.columns(2)
    if b1.button("🔄 Reset Progress", use_container_width=True):
        st.session_state.status         = {}
        st.session_state.idx            = 0
        st.session_state.flip           = False
        st.session_state.quiz_completed = False
        st.session_state.start_time     = time.time()
        st.rerun()

    if b2.button("📁 New Topic", use_container_width=True):
        for key in ["cards", "status", "quiz_completed"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.idx   = 0
        st.session_state.flip  = False
        st.rerun()
