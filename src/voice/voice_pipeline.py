"""Voice Pipeline for ZentroMart VoiceOps using Gnani Speech Models.

Features:
- Speech-to-Text (STT) via GnaniSTTClient
- Text-to-Speech (TTS) via GnaniTTSClient (model="timbre-v2.5", voice="Kaveri")
- Text normalization for the ear:
  - Spaced out digits for order IDs and phone digits (e.g. "8 9 7 8, 2 8 3 7")
  - Rupee symbol replacement ("rupees" / "रुपये")
  - Clean pronunciation of dates and abbreviations
- Graceful recovery on silent, empty, or unclear audio
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import re
from dotenv import load_dotenv

load_dotenv()

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

def normalize_text_for_speech(text: str, language: str = "en") -> str:
    """Normalizes text for natural TTS pronunciation (ear-friendly)."""
    if not text:
        return ""

    out = text

    # 1. Currency symbol replacement
    if language == "hi":
        out = re.sub(r"₹\s*(\d+)", r"\1 रुपये", out)
        out = out.replace("₹", " रुपये ")
    else:
        out = re.sub(r"₹\s*(\d+)", r"\1 rupees", out)
        out = out.replace("₹", " rupees ")

    # 2. Format 8-digit order IDs with spaces and pause for TTS
    def _space_order_id(match):
        digits = match.group(0)
        return f"{digits[0]} {digits[1]} {digits[2]} {digits[3]}, {digits[4]} {digits[5]} {digits[6]} {digits[7]}"

    out = re.sub(r"\b\d{8}\b", _space_order_id, out)

    # 3. Format 4-digit verification suffixes with spaces
    def _space_4_digits(match):
        d = match.group(0)
        return f"{d[0]} {d[1]} {d[2]} {d[3]}"

    # Match standalone 4 digits that are clearly numbers
    # (avoid years like 2026)
    out = re.sub(r"\b(?!(?:202[0-9]|19[0-9]{2}))\d{4}\b", _space_4_digits, out)

    # 4. Remove Markdown bold/italics formatting
    out = re.sub(r"[*_~`#]", "", out)

    # 5. Normalize whitespace
    out = re.sub(r"\s+", " ", out).strip()

    return out

class VoicePipeline:
    def __init__(self, voice: str = "Kaveri", model: str = "timbre-v2.5"):
        self.api_key = os.getenv("GNANI_API_KEY")
        if not self.api_key:
            raise ValueError("GNANI_API_KEY is not set in environment or .env file.")
        self.stt = GnaniSTTClient()
        self.tts = GnaniTTSClient()
        self.voice = voice
        self.model = model

    def transcribe(self, audio_file_path: str, language_code: str = "en-IN") -> str:
        """Transcribe an audio file using Gnani STT."""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            res = self.stt.transcribe(audio_file_path, language_code=language_code)
            if isinstance(res, dict):
                return res.get("transcript", "").strip()
            return str(res).strip()
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""

    def synthesize(self, text: str, output_path: str, language_code: str = "en-IN") -> str:
        """Synthesize text into speech (.wav) using Gnani TTS."""
        lang_group = "hi" if language_code.startswith("hi") else "en"
        speech_text = normalize_text_for_speech(text, language=lang_group)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            self.tts.synthesize(
                speech_text,
                voice=self.voice,
                model=self.model,
                language=language_code,
                output_file=output_path
            )
            return output_path
        except Exception as e:
            print(f"[TTS Error] {e}")
            return ""

    def voice_turn(self, audio_path: str, agent, conversation_id: str, output_audio_path: str) -> Dict:
        """Complete voice turn: Audio -> STT -> Agent -> TTS -> Audio."""
        heard = self.transcribe(audio_path, language_code="en-IN")

        if not heard or not heard.strip():
            reply = "Sorry, I didn't catch that. Could you please say it again?"
            decision = "clarify"
            chunks = []
            tools = []
        else:
            agent_res = agent.process_turn(conversation_id, heard)
            reply = agent_res.get("reply", "")
            decision = agent_res.get("decision", "answer")
            chunks = agent_res.get("chunks", [])
            tools = agent_res.get("tool_calls", [])

        # Synthesize reply
        self.synthesize(reply, output_audio_path, language_code="en-IN")

        return {
            "transcript": heard,
            "reply_text": reply,
            "audio_output": output_audio_path,
            "decision": decision,
            "chunks_used": chunks,
            "tools_called": tools
        }
