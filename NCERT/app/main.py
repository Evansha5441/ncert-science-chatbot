import time
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.retriever import RAGRetriever
from app.smart_cache import (
    SmartCache,
    is_conversation_tied_request,
    contains_unresolved_followup
)
from app.llm_service import LLMService

app = FastAPI(
    title="NCERT Class 10 Science Chatbot with Smart Caching",
    description="High performance doubt-solving chatbot with multi-tier verified semantic caching.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> list of {"role": "user"|"assistant", "content": str}
sessions: Dict[str, List[Dict[str, str]]] = {}

# Lazy loaded components
retriever = None
smart_cache = None
llm_service = None

@app.on_event("startup")
def startup_event():
    global retriever, smart_cache, llm_service
    retriever = RAGRetriever.get_instance()
    smart_cache = SmartCache.get_instance()
    llm_service = LLMService()
    print("FastAPI services initialized!")

class SessionResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    citations: List[str]
    cache_hit: bool
    latency_ms: int

@app.get("/")
def health_check():
    return {
        "status": "online",
        "app": "NCERT Class 10 Science Chatbot",
        "llm_configured": llm_service.is_configured() if llm_service else False,
        "llm_provider": settings.LLM_PROVIDER
    }

@app.post("/session", response_model=SessionResponse)
def create_session():
    """Generates a new session UUID and initializes its conversation history."""
    new_id = str(uuid.uuid4())
    sessions[new_id] = []
    return {"session_id": new_id}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    start_time = time.perf_counter()

    if req.session_id not in sessions:
        sessions[req.session_id] = []

    history = sessions[req.session_id]
    user_query = req.message.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # -------------------------------------------------------------
    # STEP 1: Direct Cache Check (Strict Zero-LLM Fast Path)
    # Fast path: < 50ms, no LLM invoked, verified numeric and entity guard
    # -------------------------------------------------------------
    cached_result = smart_cache.lookup(user_query)
    if cached_result is not None:
        reply, citations = cached_result
        # Record into session history
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": reply})

        latency = int((time.perf_counter() - start_time) * 1000)
        return ChatResponse(
            reply=reply,
            citations=citations,
            cache_hit=True,
            latency_ms=latency
        )

    # -------------------------------------------------------------
    # STEP 2: Cache Miss / Follow-up Processing
    # If the user query is a pronoun follow-up (e.g. "what about its laws?"),
    # we rewrite it using conversation history to get the standalone topic.
    # -------------------------------------------------------------
    is_tied = is_conversation_tied_request(user_query)
    standalone_query = user_query

    if contains_unresolved_followup(user_query) and history:
        standalone_query = llm_service.rewrite_query_for_retrieval(user_query, history)
        print(f"Follow-up detected: '{user_query}' -> Rewritten as: '{standalone_query}'")

        # Now check if the rewritten canonical query exists in cache!
        # Notice: When a student asks "What about its laws?" after "What is refraction?",
        # the rewritten query "What are the laws of refraction of light?" might ALREADY be in cache!
        if not is_tied:
            cached_rewrite = smart_cache.lookup(standalone_query)
            if cached_rewrite is not None:
                reply, citations = cached_rewrite
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                latency = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    reply=reply,
                    citations=citations,
                    cache_hit=True,
                    latency_ms=latency
                )

    # -------------------------------------------------------------
    # STEP 3: RAG Retrieval from NCERT FAISS Vector Store
    # -------------------------------------------------------------
    context, citations, scores = retriever.retrieve(standalone_query)

    # Out of syllabus guard: if closest similarity distance is very poor
    # MiniLM L6 v2 L2 distance typically ranges 0.3 - 1.2 for related text.
    # > 1.45 means completely unrelated to Class 10 Science (e.g. quantum chromodynamics, movie gossip)
    min_distance = min(scores) if scores else 999.0
    if min_distance > 1.48:
        reply = "I am designed to answer doubts from the NCERT Class 10 Science textbook. This topic is not covered in the book."
        citations = []
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": reply})
        latency = int((time.perf_counter() - start_time) * 1000)
        return ChatResponse(
            reply=reply,
            citations=citations,
            cache_hit=False,
            latency_ms=latency
        )

    # -------------------------------------------------------------
    # STEP 4: LLM Generation
    # -------------------------------------------------------------
    reply = llm_service.generate_answer(
        query=user_query,
        context=context,
        citations=citations,
        history=history
    )

    # -------------------------------------------------------------
    # STEP 5: Store into Smart Cache (with Safety Rules)
    # We only cache if:
    # 1. Not conversation-tied ("Explain it more simply")
    # 2. We use the standalone query so future queries benefit
    # 3. Topic was covered and answered safely
    # -------------------------------------------------------------
    if not is_tied and citations and "not covered in the syllabus" not in reply.lower():
        smart_cache.store(standalone_query, reply, citations)

    # Update session history
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": reply})

    latency = int((time.perf_counter() - start_time) * 1000)
    return ChatResponse(
        reply=reply,
        citations=citations,
        cache_hit=False,
        latency_ms=latency
    )
