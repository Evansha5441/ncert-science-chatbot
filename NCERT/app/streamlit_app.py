import time
import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="NCERT Class 10 Science Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .cache-hit {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #10B981;
    }
    .cache-miss {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #3B82F6;
    }
    .chapter-tag {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #F59E0B;
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration & Backend URL
API_URL = st.sidebar.text_input("FastAPI Backend URL (optional)", value="http://127.0.0.1:8000")

# Lazy-loaded direct in-process services for standalone Streamlit Cloud hosting
@st.cache_resource(show_spinner="Loading NCERT Knowledge Base & Smart Cache...")
def load_inprocess_services():
    from app.retriever import RAGRetriever
    from app.smart_cache import SmartCache
    from app.llm_service import LLMService
    retriever = RAGRetriever.get_instance()
    cache = SmartCache.get_instance()
    llm = LLMService()
    return retriever, cache, llm

def process_query_inprocess(user_query: str, history: list):
    import time
    from app.smart_cache import is_conversation_tied_request, contains_unresolved_followup
    retriever, smart_cache, llm_service = load_inprocess_services()
    start_time = time.perf_counter()

    # Step 1: Direct Cache Check
    cached_result = smart_cache.lookup(user_query)
    if cached_result is not None:
        reply, citations = cached_result
        latency = int((time.perf_counter() - start_time) * 1000)
        return reply, citations, True, latency

    # Step 2: Follow-up query rewriting
    is_tied = is_conversation_tied_request(user_query)
    standalone_query = user_query
    if contains_unresolved_followup(user_query) and history:
        standalone_query = llm_service.rewrite_query_for_retrieval(user_query, history)
        if not is_tied:
            cached_rewrite = smart_cache.lookup(standalone_query)
            if cached_rewrite is not None:
                reply, citations = cached_rewrite
                latency = int((time.perf_counter() - start_time) * 1000)
                return reply, citations, True, latency

    # Step 3: Retrieval
    context, citations, scores = retriever.retrieve(standalone_query)
    min_dist = min(scores) if scores else 999.0
    if min_dist > 1.48:
        reply = "I am designed to answer doubts from the NCERT Class 10 Science textbook. This topic is not covered in the book."
        latency = int((time.perf_counter() - start_time) * 1000)
        return reply, [], False, latency

    # Step 4: LLM Generation
    reply = llm_service.generate_answer(
        query=user_query,
        context=context,
        citations=citations,
        history=history
    )

    # Step 5: Store into Cache
    if not is_tied and citations and "not covered in the syllabus" not in reply.lower():
        smart_cache.store(standalone_query, reply, citations)

    latency = int((time.perf_counter() - start_time) * 1000)
    return reply, citations, False, latency

def get_new_session(api_url: str):
    try:
        res = requests.post(f"{api_url}/session", timeout=2)
        if res.status_code == 200:
            return res.json().get("session_id")
    except Exception:
        pass
    import uuid
    return str(uuid.uuid4())


# Session State initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = get_new_session(API_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls & Analytics
st.sidebar.title("🔬 Prepzy AI Doubt-Solver")
st.sidebar.markdown("**Class 10 Science Textbook Assistant**")

if st.sidebar.button("🔄 New Conversation", use_container_width=True):
    st.session_state.session_id = get_new_session(API_URL)
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`" if st.session_state.session_id else "**Offline**")

# Analytics calculation
total_turns = len(st.session_state.messages) // 2
hits = sum(1 for m in st.session_state.messages if m.get("cache_hit") is True)
hit_rate = (hits / total_turns * 100) if total_turns > 0 else 0.0

st.sidebar.metric(label="Total Queries", value=total_turns)
col_a, col_b = st.sidebar.columns(2)
col_a.metric(label="Cache Hits", value=hits)
col_b.metric(label="Hit Rate", value=f"{hit_rate:.0f}%")

st.sidebar.divider()
st.sidebar.markdown("""
### 🧠 Cache Rules in Action:
- **Same doubt, diff words:** Cache hit (Instant, 0 LLM tokens)
- **Concave vs Convex:** Cache miss (Guards prevent wrong physics)
- **R = 20 cm vs 30 cm:** Cache miss (Guards prevent number collision)
- **'Explain simply':** Cache miss (Preserves adaptive learning)
""")

# Main Chat Header
st.markdown('<div class="main-title">NCERT Class 10 Science Doubt-Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instant, accurate answers backed by verified textbook chapters with Smart Caching</div>', unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Display metadata badges
            cols = st.columns([1, 4])
            with cols[0]:
                if msg.get("cache_hit"):
                    st.markdown(f'<span class="metric-badge cache-hit">⚡ Cache Hit ({msg.get("latency_ms", 0)} ms)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="metric-badge cache-miss">🤖 LLM Fresh ({msg.get("latency_ms", 0)} ms)</span>', unsafe_allow_html=True)
            with cols[1]:
                citations = msg.get("citations", [])
                if citations:
                    citation_html = " ".join([f'<span class="chapter-tag">📖 {c}</span>' for c in citations])
                    st.markdown(citation_html, unsafe_allow_html=True)

# Chat Input Box
user_prompt = st.chat_input("Ask any doubt from NCERT Class 10 Science (e.g., 'What is refraction?')...")

if user_prompt:
    if not st.session_state.session_id:
        st.session_state.session_id = get_new_session(API_URL)

    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Query Backend or Fallback In-Process
    with st.chat_message("assistant"):
        with st.spinner("Finding answer from NCERT textbook..."):
            reply, citations, cache_hit, latency_ms = None, [], False, 0
            # Try FastAPI backend first
            try:
                payload = {
                    "session_id": st.session_state.session_id or "default-session",
                    "message": user_prompt
                }
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    reply = data["reply"]
                    citations = data.get("citations", [])
                    cache_hit = data.get("cache_hit", False)
                    latency_ms = data.get("latency_ms", 0)
            except Exception:
                pass

            # If backend is not running or unreachable (e.g. running standalone on Streamlit Cloud)
            if reply is None:
                reply, citations, cache_hit, latency_ms = process_query_inprocess(
                    user_prompt,
                    st.session_state.messages
                )

            st.markdown(reply)
            
            # Display metadata badges
            cols = st.columns([1, 4])
            with cols[0]:
                if cache_hit:
                    st.markdown(f'<span class="metric-badge cache-hit">⚡ Cache Hit ({latency_ms} ms)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="metric-badge cache-miss">🤖 LLM Fresh ({latency_ms} ms)</span>', unsafe_allow_html=True)
            with cols[1]:
                if citations:
                    citation_html = " ".join([f'<span class="chapter-tag">📖 {c}</span>' for c in citations])
                    st.markdown(citation_html, unsafe_allow_html=True)

            # Save to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "citations": citations,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms
            })

