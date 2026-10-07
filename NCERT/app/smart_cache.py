import os
import json
import sqlite3
import re
import numpy as np
    try:
        import faiss
        HAS_FAISS = True
    except ImportError:
        HAS_FAISS = False
        faiss = None

from typing import Optional, Tuple, List, Dict
from sentence_transformers import SentenceTransformer
from app.config import settings

# Key structural entities that differentiate questions even if words are semantically close
KEY_ENTITIES = [
    # Optical surfaces & lenses
    "concave mirror", "convex mirror", "concave lens", "convex lens", "plane mirror",
    # Electric circuits
    "series", "parallel", "ammeter", "voltmeter", "fuse",
    # Respiration & metabolism
    "aerobic", "anaerobic", "photosynthesis", "respiration",
    # Fission & fusion / cells
    "binary fission", "multiple fission", "mitosis", "meiosis",
    # Chemical reaction types
    "endothermic", "exothermic", "combination", "decomposition", "displacement", "double displacement", "redox", "oxidation", "reduction",
    # Metals / chemistry
    "acid", "base", "salt", "metal", "non-metal", "saponification", "esterification",
    # Genes
    "dominant", "recessive", "genotype", "phenotype"
]

# Patterns representing conversation-tied requests that MUST NOT be served from or stored in global cache
CONVERSATION_TIED_PATTERNS = [
    r"\bexplain\s+(it\s+)?(more\s+)?simply\b",
    r"\bsimplif(y|ied)\b",
    r"\bin\s+simpler\s+terms\b",
    r"\bin\s+simple\s+words\b",
    r"\bexplain\s+like\s+i('?m|\s+am)\s+5\b",
    r"\belaborate(\s+more)?\b",
    r"\bshorter\b",
    r"\bbriefer\b",
    r"\bsummarize\s+(that|this|it)\b",
    r"\bexpand\s+on\s+(that|this|it)\b",
    r"\bgive\s+(me\s+)?an?\s+analogy\b",
    r"\brepeat\s+that\b",
    r"\bmore\s+detail(s)?\b"
]

# Pronoun / follow-up patterns indicating query is not standalone
FOLLOWUP_PRONOUNS = [
    r"\bits\s+[a-z]",
    r"\bwhat\s+about\s+(it|its|them|this|that|these|those)\b",
    r"\bwhy\s+is\s+(it|this|that)\b",
    r"\bhow\s+does\s+(it|this|that)\b",
    r"\band\s+(its|their|the)\b",
    r"\bwhat\s+is\s+its\b",
    r"\btell\s+me\s+about\s+it\b"
]

def extract_numbers_and_units(text: str) -> set:
    """Extract numbers with optional units (e.g., 20 cm, 30 cm, 5 A, 10 V)."""
    # Find numbers (integers, floats) along with optional units
    matches = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:cm|m|mm|v|a|w|ohm|hz|j|kg|g|s|min|h|°c|k))?\b', text.lower())
    # Normalize spaces inside units
    return {re.sub(r'\s+', '', m) for m in matches}

def extract_key_entities(text: str) -> set:
    """Extract critical domain entities present in the query."""
    text_lower = text.lower()
    entities = set()
    for ent in KEY_ENTITIES:
        if ent in text_lower:
            entities.add(ent)
    return entities

def is_conversation_tied_request(text: str) -> bool:
    """Detect requests that are inherently dependent on the previous assistant response style/depth."""
    text_lower = text.lower().strip()
    for pattern in CONVERSATION_TIED_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def contains_unresolved_followup(text: str) -> bool:
    """Detect unresolved pronoun follow-ups like 'what about its laws?'."""
    text_lower = text.lower().strip()
    for pattern in FOLLOWUP_PRONOUNS:
        if re.search(pattern, text_lower):
            return True
    return False

class SmartCache:
    _instance = None

    def __init__(self, db_path: str = None, dimension: int = 384):
        self.db_path = db_path or settings.CACHE_DB_PATH
        self.dimension = dimension
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self._init_db()
        self._load_cache_index()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SmartCache()
        return cls._instance

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    canonical_query TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    numbers_json TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def _load_cache_index(self):
        """Build FAISS IndexFlatIP (cosine similarity on normalized vectors) from SQLite."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.cache_records: List[Dict] = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, query, canonical_query, reply, citations, numbers_json, entities_json FROM semantic_cache ORDER BY id ASC')
            rows = cursor.fetchall()

        if rows:
            queries = [r[2] for r in rows]  # embed canonical queries
            embeddings = self.embed_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
            self.index.add(embeddings.astype(np.float32))

            for r in rows:
                self.cache_records.append({
                    "id": r[0],
                    "query": r[1],
                    "canonical_query": r[2],
                    "reply": r[3],
                    "citations": json.loads(r[4]),
                    "numbers": set(json.loads(r[5])),
                    "entities": set(json.loads(r[6]))
                })
        print(f"SmartCache initialized with {len(self.cache_records)} cached entries.")

    def lookup(self, query: str) -> Optional[Tuple[str, List[str]]]:
        """
        Check if query can be served from cache safely.
        Returns:
            (reply, citations) if cache hit
            None if cache miss
        STRICT RULES:
        1. Conversation-tied requests ("Explain it more simply") -> NEVER served from cache
        2. Unresolved pronoun follow-ups ("What about its laws?") -> NEVER served from cache directly
        3. Numbers & units mismatch -> REJECT cache hit
        4. Key contrasting entity mismatch (concave vs convex, series vs parallel) -> REJECT cache hit
        5. Semantic similarity must exceed strict threshold (>= 0.91)
        """
        if is_conversation_tied_request(query):
            return None

        if contains_unresolved_followup(query):
            return None

        if not self.cache_records or self.index.ntotal == 0:
            return None

        # Extract features of the incoming query
        query_numbers = extract_numbers_and_units(query)
        query_entities = extract_key_entities(query)

        # Compute embedding and search
        query_vec = self.embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        # Search top 3 nearest candidates
        k = min(3, self.index.ntotal)
        sims, indices = self.index.search(query_vec, k)

        for score, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self.cache_records):
                continue

            candidate = self.cache_records[idx]

            # Rule 5: Semantic cosine threshold
            if score < settings.CACHE_SIMILARITY_THRESHOLD:
                continue

            # Rule 3: Exact number / unit verification (e.g. 20 cm vs 30 cm)
            if query_numbers != candidate["numbers"]:
                # Numbers differ!
                continue

            # Rule 4: Contrasting entity verification (e.g. concave mirror vs convex mirror)
            if query_entities != candidate["entities"]:
                # Entities differ!
                continue

            # Verified safe cache hit!
            return candidate["reply"], candidate["citations"]

        return None

    def store(self, query: str, reply: str, citations: List[str]):
        """
        Store answered question into cache ONLY if it is safe and reusable.
        Never store conversation-tied requests ("Explain more simply").
        Never store unresolved pronoun follow-ups ("What about its laws?").
        Never store declined questions (e.g. out of syllabus).
        """
        # Guard 1: Do not cache out-of-syllabus declines or conversational meta replies
        if not citations or "not cover" in reply.lower() or "outside the scope" in reply.lower():
            return

        # Guard 2: Do not cache conversational styles
        if is_conversation_tied_request(query):
            return

        # Guard 3: Do not cache unresolved pronouns
        if contains_unresolved_followup(query):
            return

        query_numbers = extract_numbers_and_units(query)
        query_entities = extract_key_entities(query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO semantic_cache (query, canonical_query, reply, citations, numbers_json, entities_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                query,
                query.strip(),
                reply,
                json.dumps(citations),
                json.dumps(list(query_numbers)),
                json.dumps(list(query_entities))
            ))
            conn.commit()
            record_id = cursor.lastrowid

        # Add to in-memory FAISS index
        query_vec = self.embed_model.encode([query.strip()], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        self.index.add(query_vec)

        self.cache_records.append({
            "id": record_id,
            "query": query,
            "canonical_query": query.strip(),
            "reply": reply,
            "citations": citations,
            "numbers": query_numbers,
            "entities": query_entities
        })
