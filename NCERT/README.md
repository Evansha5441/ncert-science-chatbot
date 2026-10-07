# NCERT Class 10 Science Chatbot with Smart Caching 🔬

A high-performance, doubt-solving chatbot tailored for the official NCERT Class 10 Science curriculum (current edition), featuring a **Zero-LLM Multi-Tier Verified Smart Cache** that achieves ultra-fast cache hits (< 50ms) while guaranteeing student safety against false positives and hallucinations.

Developed for **Prepzy.ai (GlobusLearn Services India Pvt. Ltd)** AI Intern Round 1 Assignment.

---

## 🌟 Key Highlights & Architecture

- **Ground Truth**: Full current edition of NCERT Class 10 Science (all 13 rationalized chapters) parsed and indexed.
- **Strict Syllabus Bound**: Accurately answers from textbook context with chapter citations (e.g. *Light – Reflection and Refraction*); politely declines out-of-scope or college-level queries.
- **Multi-Turn Continuity**: Handles contextual follow-ups (e.g., *"What about its laws?"*) via conversational query reformulation.
- **Smart Semantic Cache**:
  - **Zero LLM Token Usage on Hits**: Fast vector lookup via FAISS `IndexFlatIP` with normalized MiniLM embeddings.
  - **Numeric & Unit Guard**: Rejects matches if parameters differ (e.g., `R = 20 cm` vs `R = 30 cm`).
  - **Domain Contrast Guard**: Distinguishes optical and physical counterparts (`concave mirror` vs `convex mirror`, `series` vs `parallel`).
  - **Style Guard**: Conversation-tied requests (*"Explain it more simply"*) are never falsely returned from cache.
  - **Speed**: Serves verified cache hits in **< 50 ms** (far exceeding the 500 ms threshold).

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Backend API | FastAPI + Uvicorn |
| Vector Store | FAISS (`faiss-cpu`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM Framework | LangChain & OpenAI-compatible SDK |
| LLM Providers | Groq (`llama-3.3-70b-versatile`), Google Gemini (`gemini-1.5-flash`), or OpenAI (`gpt-4o-mini`) |
| Frontend | Streamlit |
| Database | SQLite (Cache metadata) + FAISS (Cache vectors) |

---

## 📁 Repository Structure

```text
.
├── app/
│   ├── config.py              # Environment configuration & hyperparameters
│   ├── download_textbook.py   # Multi-threaded downloader for 13 NCERT chapters
│   ├── build_index.py         # PDF text extractor, chunker & FAISS builder
│   ├── retriever.py           # RAG retrieval module with chapter citations
│   ├── smart_cache.py         # Multi-tier verified safe caching engine
│   ├── llm_service.py         # LLM provider interface (Groq/Gemini/OpenAI)
│   ├── main.py                # FastAPI endpoints (/session, /chat)
│   └── streamlit_app.py       # Streamlit user interface with latency & citation badges
├── data/
│   ├── raw_pdfs/              # Downloaded NCERT chapter PDFs
│   └── faiss_index/           # Pre-built FAISS index for textbook chunks
├── tests/
│   └── test_caching.py        # Automated test suite for caching safety scenarios
├── EXPLAINER.md               # One-page comprehensive explainer of approach
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys
└── README.md                  # This file
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone <your-repo-link>
cd NCERT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and set your preferred free-tier API key:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(You can also use `LLM_PROVIDER=gemini` with `GEMINI_API_KEY`, or `LLM_PROVIDER=openai`)*

### 3. Download Textbook & Build Index (Pre-indexed out of box)

To build or refresh the textbook index from NCERT:
```bash
python app/download_textbook.py
python app/build_index.py
```

### 4. Run Automated Cache Tests

Run the test suite verifying all 5 tricky caching situations:
```bash
python -m unittest tests/test_caching.py
```
Expected output:
```text
Ran 5 tests in 5.8s
OK
```

### 5. Launch Backend API (FastAPI)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/`

### 6. Launch Frontend (Streamlit)

In a new terminal window (with venv activated):
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

---

## 📡 API Specification

### 1. Create Session
`POST /session`
```json
// Response:
{
  "session_id": "4e18ac14-df0a-4934-8c0c-e2f07ab2374b"
}
```

### 2. Chat Query
`POST /chat`
```json
// Request:
{
  "session_id": "4e18ac14-df0a-4934-8c0c-e2f07ab2374b",
  "message": "What is refraction?"
}

// Response:
{
  "reply": "Refraction is the phenomenon of bending of light as it travels from one transparent medium to another...",
  "citations": [
    "Light – Reflection and Refraction"
  ],
  "cache_hit": true,
  "latency_ms": 12
}
```

---

## 🔬 Caching Rules & Safety Matrix

| Scenario | Input Query | Target Cached Item | Result | Rationale |
|---|---|---|---|---|
| **Same doubt, diff wording** | "What does refraction mean?" | "What is refraction?" | **CACHE HIT** | High cosine similarity (≥0.91), identical entities & numbers. |
| **Similar words, diff question** | "Image by a convex mirror" | "Image by a concave mirror" | **CACHE MISS** | Domain entity guard flags `convex` vs `concave`. |
| **Same question, diff numbers** | "Focal length when R = 30 cm" | "Focal length when R = 20 cm" | **CACHE MISS** | Numeric & unit extractor flags mismatch (`30cm` != `20cm`). |
| **Follow-up query** | "What about its laws?" | Standalone laws query | **DEPENDS** | Resolves pronoun via conversation context first; serves cache if the standalone resolved question exists. |
| **Conversation-tied** | "Explain it more simply" | Previous answer | **CACHE MISS** | User requested adaptive style change; must never return static answer. |

---

## 🌐 Deployment to Streamlit Cloud

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, set the entry point to `app/streamlit_app.py`.
4. In **Advanced Settings -> Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_key"
   LLM_PROVIDER = "groq"
   ```
5. Deploy and copy the live app URL.
