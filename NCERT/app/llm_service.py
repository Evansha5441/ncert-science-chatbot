import os
from typing import List, Dict, Optional
from openai import OpenAI
from app.config import settings

class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.client = None
        self._init_client()

    def _get_secret(self, key: str, default: str = "") -> str:
        # Check Streamlit secrets first, then env/settings
        try:
            import streamlit as st
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
        return getattr(settings, key, "") or os.environ.get(key, default)

    def _init_client(self):
        self.provider = self._get_secret("LLM_PROVIDER", "groq").lower()

        if self.provider == "groq" or (not self.client and self._get_secret("GROQ_API_KEY")):
            api_key = self._get_secret("GROQ_API_KEY")
            if api_key:
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key
                )
                self.model = self._get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")
                self.provider = "groq"
                return

        if self.provider == "gemini" or (not self.client and self._get_secret("GEMINI_API_KEY")):
            api_key = self._get_secret("GEMINI_API_KEY")
            if api_key:
                # Gemini OpenAI compatible endpoint
                self.client = OpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=api_key
                )
                self.model = self._get_secret("GEMINI_MODEL", "gemini-1.5-flash")
                self.provider = "gemini"
                return

        if self.provider == "openai" or (not self.client and self._get_secret("OPENAI_API_KEY")):
            api_key = self._get_secret("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model = self._get_secret("OPENAI_MODEL", "gpt-4o-mini")
                self.provider = "openai"
                return


    def is_configured(self) -> bool:
        return self.client is not None

    def rewrite_query_for_retrieval(self, message: str, history: List[Dict[str, str]]) -> str:
        """
        Rewrites conversational follow-up questions (e.g. 'what about its laws?')
        into a standalone self-contained query using recent conversation turns.
        """
        if not history or not self.client:
            return message

        # Only prompt rewrite if pronouns or conversational indicators are present
        pronoun_words = ["it", "its", "they", "them", "this", "that", "these", "those", "laws", "formula", "example", "why", "how", "difference"]
        words = set(message.lower().split())
        if not (words & set(pronoun_words)) and len(message.split()) > 4:
            return message

        recent_history = history[-4:]
        history_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in recent_history])

        prompt = f"""Given this conversation history between a student and an NCERT science tutor:
{history_str}

Follow-up student query: "{message}"

Rewrite the student's follow-up into a single standalone scientific query that incorporates the context of earlier turns.
Keep it strictly concise and return ONLY the rewritten standalone query, nothing else."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            rewritten = resp.choices[0].message.content.strip().strip('"')
            return rewritten if rewritten else message
        except Exception as e:
            print(f"Query rewrite error: {e}")
            return message

    def generate_answer(
        self,
        query: str,
        context: str,
        citations: List[str],
        history: List[Dict[str, str]]
    ) -> str:
        """
        Generates an accurate answer strictly based on the textbook context.
        Politely declines out-of-scope topics.
        """
        if not self.client:
            # Fallback mock for offline demonstration / testing
            return (
                f"Based on the NCERT Class 10 Science textbook ({', '.join(citations)}):\n\n"
                f"Here is information regarding your query:\n"
                f"{context[:400]}...\n\n"
                f"(Note: To enable live LLM synthesis, configure GROQ_API_KEY or GEMINI_API_KEY in .env)."
            )

        system_prompt = f"""You are an expert, encouraging AI Doubt-Solving Tutor for Indian students studying NCERT Class 10 Science.

STRICT INSTRUCTIONS:
1. Answer the student's doubt using ONLY the provided NCERT textbook excerpts.
2. If the question asks about topics not covered in NCERT Class 10 Science (e.g. quantum mechanics, general knowledge, movies, higher grade college physics, or unrelated subjects), politely decline by stating: "I am designed to answer doubts from the NCERT Class 10 Science textbook. This topic is not covered in the syllabus."
3. At the end of every valid answer, cite the chapter name(s) you used in format: "Chapter: <Chapter Name>".
4. Explain clearly, accurately, with step-by-step reasoning or bullet points when helpful for a Class 10 student.
5. If the student asks you to simplify or rephrase ("explain it simply"), adapt your tone and explanations without changing the underlying scientific facts.

TEXTBOOK CONTEXT EXCERPTS:
{context}
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation turns
        for turn in history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
            
        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with LLM ({self.provider}): {str(e)}"
