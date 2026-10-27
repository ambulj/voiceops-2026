"""Ollama LLM Client for ZentroMart VoiceOps.

Implements:
- Step 1: Prompt-only baseline generation (no KB access)
- Step 2: Grounded generation with strict KB chunk constraints & unanswerable fallback
- Multi-lingual generation (English, Hindi, Hinglish)
- System instructions ensuring role clarity, AI assistant disclosure, and factual grounding
"""
from typing import Dict, List, Optional
import json
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

SYSTEM_PROMPT_STEP1 = """You are ZentroMart's AI customer support assistant.
ZentroMart is an Indian online store selling groceries, electronics, and fashion.
Always introduce yourself as an AI assistant in your reply.
Keep your response short, polite, and helpful (2-3 sentences).
Reply in the same language as the customer (English, Hindi, or Hinglish).
"""

SYSTEM_PROMPT_STEP2 = """You are ZentroMart's AI customer support assistant.
Your answers must be strictly grounded ONLY in the provided Knowledge Base context.
Rules:
1. Always state you are ZentroMart's AI assistant.
2. If the context does not contain enough information to answer the question, do NOT invent facts, dates, amounts, or policies. State: "Sorry, I don't have information about that. I can connect you to a colleague who can help." (In Hindi: "माफ़ कीजिए, मेरे पास इसकी जानकारी नहीं है। मैं आपको अपने सहयोगी से कनेक्ट कर सकता हूँ।")
3. Follow policy precedence: policy overrides FAQ. For ZentroMart Wallet refunds, it is instant within 2 hours. For UPI, it is 1 to 3 business days.
4. Business days are Monday to Saturday, excluding warehouse holidays. Return windows use calendar days. Today's date is 20 October 2026.
5. Keep your response concise (2-3 sentences), ear-friendly, and polite.
6. Reply in the same language as the customer (English, Hindi, or Hinglish).
"""

class LLMClient:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or OLLAMA_MODEL
        self.host = (host or OLLAMA_HOST).rstrip("/")

    def _call_ollama(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 300,
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("response", "").strip()
        except Exception as e:
            return f"Error contacting LLM: {str(e)}"

    def generate_step1_baseline(self, question: str, language: str = "en") -> str:
        """Step 1: Baseline answer with no knowledge base access."""
        prompt = f"Customer question ({language}): {question}\n\nYour reply as ZentroMart support:"
        return self._call_ollama(SYSTEM_PROMPT_STEP1, prompt, temperature=0.3)

    def generate_step2_grounded(self, question: str, context_chunks: List[str], is_answerable: bool = True, language: str = "en") -> str:
        """Step 2: Grounded answer from retrieved knowledge chunks."""
        if not is_answerable or not context_chunks:
            if language == "hi":
                return "नमस्ते! मैं ZentroMart का एआई असिस्टेंट हूँ। माफ़ कीजिए, मेरे पास इस बारे में कोई जानकारी नहीं है। क्या मैं आपको हमारे सहयोगी से कनेक्ट कर दूँ?"
            elif language == "hi-en":
                return "Hello! Main ZentroMart ka AI assistant hoon. Sorry, mere paas iski jaankari nahi hai. Main aapko ek human colleague se connect kar deta hoon."
            else:
                return "Hello! I am ZentroMart's AI assistant. Sorry, I don't have information about that. I can connect you to a colleague who can help."

        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = (
            f"Context from ZentroMart documents:\n{context_str}\n\n"
            f"Customer question ({language}): {question}\n\n"
            f"Instructions:\n"
            f"- Answer based ONLY on the context above. If not covered, say you do not have that information.\n"
            f"- If the question asks about a specific document section or rule code (e.g. STY-1, STY-2), explicitly explain what that section states and requires.\n"
            f"- Keep your response concise (2-3 sentences), helpful, and polite."
        )
        return self._call_ollama(SYSTEM_PROMPT_STEP2, prompt, temperature=0.0)
