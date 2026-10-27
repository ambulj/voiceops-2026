#!/usr/bin/env python3
"""ZentroMart VoiceOps: Official Batch & Evaluation Runner.

Usage:
  1. Batch Audio Run (Official Organizer Evaluation):
     python run.py --input_dir <path_to_audio_folder> --output_dir <path_to_output_folder>

  2. Full Benchmark Evaluation (Step 1 vs Step 2 + Step 3 Conversations):
     python run.py --eval

  3. Interactive Console Chat:
     python run.py --interactive
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from src.agent.conversation import ConversationAgent
from src.agent.llm_client import LLMClient
from src.rag.retriever import PrecedenceRetriever
from src.voice.voice_pipeline import VoicePipeline

def run_batch_audio(input_dir: str, output_dir: str):
    """Processes a folder of customer audio files, saving replies (text + audio) and retrieved chunks."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(in_path.glob("*.wav")) + list(in_path.glob("*.mp3")) + list(in_path.glob("*.m4a")))
    if not audio_files:
        print(f"[!] No audio files found in directory: {input_dir}")
        return

    print(f"[*] Starting Batch Audio Processing on {len(audio_files)} audio file(s)...")
    print(f"[*] Input Directory: {input_dir}")
    print(f"[*] Output Directory: {output_dir}")

    vp = VoicePipeline()
    agent = ConversationAgent()
    manifest = []

    for idx, audio_f in enumerate(audio_files, 1):
        t0 = time.time()
        base_name = audio_f.stem
        out_audio = out_path / f"{base_name}_reply.wav"
        conv_id = f"BATCH_{base_name}"

        print(f"\n--- [{idx}/{len(audio_files)}] Processing: {audio_f.name} ---")
        turn_result = vp.voice_turn(
            audio_path=str(audio_f),
            agent=agent,
            conversation_id=conv_id,
            output_audio_path=str(out_audio)
        )
        latency = round(time.time() - t0, 2)

        record = {
            "file_name": audio_f.name,
            "transcript": turn_result["transcript"],
            "reply_text": turn_result["reply_text"],
            "reply_audio_path": str(out_audio.name),
            "chunks_used": turn_result["chunks_used"],
            "tools_called": turn_result["tools_called"],
            "decision": turn_result["decision"],
            "latency_seconds": latency
        }
        manifest.append(record)

        print(f"  Transcript : \"{turn_result['transcript']}\"")
        print(f"  Reply Text : \"{turn_result['reply_text']}\"")
        print(f"  Chunks Used: {turn_result['chunks_used']}")
        print(f"  Audio Saved: {out_audio.name} ({latency}s)")

    # Save manifest as JSON and CSV
    json_path = out_path / "batch_results.json"
    csv_path = out_path / "batch_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print("\n" + "=" * 60)
    print(f"[✓] Batch processing complete! Results saved to:")
    print(f"    - JSON Manifest: {json_path}")
    print(f"    - CSV Summary  : {csv_path}")
    print(f"    - Audio Files  : {out_path}")
    print("=" * 60)

def run_evaluation():
    """Runs Step 1 vs Step 2 question benchmark and Step 3 conversation benchmark."""
    print("[*] Running Full Evaluation Benchmark...")
    from src.evaluation.eval_benchmark import run_full_benchmark
    run_full_benchmark()

def run_interactive():
    """Runs interactive console chat for real-time testing."""
    print("[*] Starting Interactive Support Console...")
    agent = ConversationAgent()
    conv_id = f"CLI_{int(time.time())}"
    print("Agent ready! Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break
            res = agent.process_turn(conv_id, user_input)
            print(f"Agent: {res['reply']}")
            if res.get("chunks"):
                print(f"   [Chunks: {res['chunks']}]")
            if res.get("tool_calls"):
                print(f"   [Tools: {res['tool_calls']}]")
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Error]: {e}")

def main():
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps Support Agent")
    parser.add_argument("--input_dir", type=str, help="Directory containing customer .wav files for batch evaluation")
    parser.add_argument("--output_dir", type=str, default="sample_outputs", help="Directory to save agent replies and manifest")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation benchmark (Step 1 vs Step 2 + Step 3)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive console chat")

    args = parser.parse_args()

    if args.input_dir:
        run_batch_audio(args.input_dir, args.output_dir)
    elif args.eval:
        run_evaluation()
    elif args.interactive:
        run_interactive()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
