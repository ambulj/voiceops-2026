"""Comprehensive Evaluation Benchmark for ZentroMart VoiceOps.

Evaluates:
1. Step 1 Baseline (Prompt-only LLM without Knowledge Base)
2. Step 2 Grounded RAG (Precedence Retriever + Grounded LLM)
3. Step 3 Multi-turn Conversation & Tools (All 23 conversations / 96 turns)
Generates metrics and comparison table for report.pdf.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import json
import os
import re
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agent.conversation import ConversationAgent
from src.agent.llm_client import LLMClient
from src.rag.retriever import PrecedenceRetriever

def check_inclusion_criteria(text: str, must_include_str: str, must_not_include_str: str) -> Tuple[bool, bool, str]:
    """Evaluates whether text satisfies must_include and must_not_include rules.
    Rules:
    - must_include: semicolon-separated clauses (all clauses must pass).
      Within each clause, slash ('/') separated alternatives (any alternative satisfies clause).
    - must_not_include: semicolon-separated forbidden strings (none must be present).
    """
    t_lower = text.lower()

    # 1. Check must_not_include
    if must_not_include_str:
        forbidden_items = [f.strip().lower() for f in must_not_include_str.split(" ; ") if f.strip()]
        for forb in forbidden_items:
            # Handle special words
            if forb in t_lower:
                return False, False, f"Violated must_not_include: '{forb}'"

    # 2. Check must_include
    if must_include_str:
        clauses = [c.strip() for c in must_include_str.split(" ; ") if c.strip()]
        for clause in clauses:
            options = [opt.strip().lower() for opt in clause.split(" / ") if opt.strip()]
            clause_met = any(opt in t_lower for opt in options)
            if not clause_met:
                return False, True, f"Missing must_include clause: '{clause}'"

    return True, True, "OK"

def run_step1_eval(questions: List[Dict], llm: LLMClient, max_q: int = 52) -> Dict:
    print(f"\n[*] Evaluating Step 1: Prompt-Only Baseline ({max_q} questions)...")
    correct = 0
    total = min(len(questions), max_q)
    results = []

    for i, q in enumerate(questions[:total], 1):
        qid = q["id"]
        question_text = q["question"]
        lang = q.get("language", "en")
        must_inc = q.get("must_include", "")
        must_not_inc = q.get("must_not_include", "")

        t0 = time.time()
        reply = llm.generate_step1_baseline(question_text, language=lang)
        lat = round(time.time() - t0, 2)

        passed, not_inc_ok, msg = check_inclusion_criteria(reply, must_inc, must_not_inc)
        if passed:
            correct += 1

        results.append({
            "id": qid,
            "question": question_text,
            "reply": reply,
            "passed": passed,
            "latency": lat,
            "status": msg
        })
        print(f"  [{i:02d}/{total}] {qid} -> {'PASS' if passed else 'FAIL'} ({lat}s)")

    acc = round((correct / total) * 100, 1)
    print(f"[✓] Step 1 Baseline Accuracy: {acc}% ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total, "results": results}

def run_step2_eval(questions: List[Dict], retriever: PrecedenceRetriever, llm: LLMClient, max_q: int = 52) -> Dict:
    print(f"\n[*] Evaluating Step 2: Grounded RAG ({max_q} questions)...")
    correct = 0
    total = min(len(questions), max_q)
    results = []
    gold_recall_hits = 0
    total_gold = 0

    for i, q in enumerate(questions[:total], 1):
        qid = q["id"]
        question_text = q["question"]
        lang = q.get("language", "en")
        must_inc = q.get("must_include", "")
        must_not_inc = q.get("must_not_include", "")
        gold_raw = q.get("gold_evidence", "")
        expected_ans = q.get("answerable") == "True"
        gold_secs = [s.strip() for s in gold_raw.split(" ; ") if s.strip()]

        t0 = time.time()
        # Retrieve chunks
        retrieved, is_ans = retriever.retrieve(
            question_text, top_k=4, entity_hint=q.get("entities")
        )
        chunk_ids = [r.section_id for r in retrieved]
        context_chunks = [f"[{r.section_id}] {r.chunk.section_title}\n{r.chunk.content}" for r in retrieved]

        # Calculate retrieval recall
        if not expected_ans:
            if len(chunk_ids) == 0:
                gold_recall_hits += 1
            total_gold += 1
        else:
            hits = sum(1 for g in gold_secs if g in chunk_ids)
            gold_recall_hits += hits
            total_gold += len(gold_secs)

        # Generate grounded reply
        reply = llm.generate_step2_grounded(
            question_text, context_chunks=context_chunks, is_answerable=is_ans, language=lang
        )
        lat = round(time.time() - t0, 2)

        passed, not_inc_ok, msg = check_inclusion_criteria(reply, must_inc, must_not_inc)
        if passed:
            correct += 1

        results.append({
            "id": qid,
            "question": question_text,
            "reply": reply,
            "chunks_used": chunk_ids,
            "passed": passed,
            "latency": lat,
            "status": msg
        })
        print(f"  [{i:02d}/{total}] {qid} -> {'PASS' if passed else 'FAIL'} ({lat}s) [Chunks: {chunk_ids}]")

    acc = round((correct / total) * 100, 1)
    retrieval_recall = round((gold_recall_hits / total_gold) * 100, 1) if total_gold else 0
    print(f"[✓] Step 2 Grounded Accuracy: {acc}% ({correct}/{total})")
    print(f"[✓] Step 2 Retrieval Recall: {retrieval_recall}% ({gold_recall_hits}/{total_gold})")
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "retrieval_recall": retrieval_recall,
        "results": results
    }

def run_step3_eval(conversations_file: str, max_turns: int = 96) -> Dict:
    print(f"\n[*] Evaluating Step 3: Multi-Turn Support Conversations...")
    with open(conversations_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    agent = ConversationAgent()
    total = min(len(rows), max_turns)
    correct = 0
    tool_matches = 0
    total_tools_expected = 0

    results = []

    for i, row in enumerate(rows[:total], 1):
        cid = row["conversation_id"]
        turn_num = int(row["turn"])
        msg = row["customer_message"]
        exp_decision = row["expected_decision"]
        must_inc = row.get("must_include", "")
        must_not_inc = row.get("must_not_include", "")
        exp_tools = row.get("expected_tool_calls", "")

        res = agent.process_turn(cid, msg)
        reply = res["reply"]
        actual_tools = res["tool_calls"]
        actual_decision = res["decision"]

        passed, _, note = check_inclusion_criteria(reply, must_inc, must_not_inc)
        
        # Tool call evaluation
        tools_ok = True
        if exp_tools:
            total_tools_expected += 1
            # Check tool presence
            expected_func = exp_tools.split("(")[0]
            if any(expected_func in act for act in actual_tools):
                tool_matches += 1
            else:
                tools_ok = False

        if passed and tools_ok:
            correct += 1

        results.append({
            "conversation_id": cid,
            "turn": turn_num,
            "user": msg,
            "agent_reply": reply,
            "decision": actual_decision,
            "expected_decision": exp_decision,
            "tools_called": actual_tools,
            "expected_tools": exp_tools,
            "passed": passed and tools_ok,
            "note": note
        })

    conv_acc = round((correct / total) * 100, 1)
    tool_acc = round((tool_matches / total_tools_expected) * 100, 1) if total_tools_expected else 100
    print(f"[✓] Step 3 Conversation Handling Accuracy: {conv_acc}% ({correct}/{total})")
    print(f"[✓] Step 3 Tool Calling Accuracy: {tool_acc}% ({tool_matches}/{total_tools_expected})")
    return {
        "accuracy": conv_acc,
        "correct": correct,
        "total": total,
        "tool_accuracy": tool_acc,
        "results": results
    }

def run_full_benchmark():
    dataset_dir = ROOT_DIR / "Capstone_Project_Dataset_FTTT_2026"
    q_file = dataset_dir / "practice" / "questions.csv"
    c_file = dataset_dir / "practice" / "conversations.csv"

    with q_file.open(encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    llm = LLMClient()
    retriever = PrecedenceRetriever()

    # Step 1 Baseline (sample 15 questions for quick comparative run or all 52)
    step1_data = run_step1_eval(questions, llm, max_q=15)

    # Step 2 Grounded RAG (sample 15 questions for quick comparative run or all 52)
    step2_data = run_step2_eval(questions, retriever, llm, max_q=15)

    # Step 3 Multi-Turn Conversations (all 23 conversations, 96 turns)
    step3_data = run_step3_eval(str(c_file), max_turns=96)

    # Save summary report
    summary = {
        "step1_baseline": {
            "questions_evaluated": step1_data["total"],
            "accuracy_percent": step1_data["accuracy"],
            "correct": step1_data["correct"]
        },
        "step2_grounded_rag": {
            "questions_evaluated": step2_data["total"],
            "accuracy_percent": step2_data["accuracy"],
            "retrieval_recall_percent": step2_data["retrieval_recall"],
            "correct": step2_data["correct"]
        },
        "step3_conversations": {
            "total_turns": step3_data["total"],
            "conversation_accuracy_percent": step3_data["accuracy"],
            "tool_calling_accuracy_percent": step3_data["tool_accuracy"],
            "correct": step3_data["correct"]
        }
    }

    report_path = ROOT_DIR / "sample_outputs" / "benchmark_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("           VOICEOPS 2026 BENCHMARK SUMMARY REPORT")
    print("=" * 60)
    print(f"Step 1 (Prompt-Only Baseline) Accuracy : {step1_data['accuracy']}%")
    print(f"Step 2 (Grounded RAG) Accuracy         : {step2_data['accuracy']}%")
    print(f"Step 2 Knowledge Retrieval Recall      : {step2_data['retrieval_recall']}%")
    print(f"Step 3 Multi-turn Conversation Accuracy: {step3_data['accuracy']}%")
    print(f"Step 3 Tool-Calling Execution Accuracy : {step3_data['tool_accuracy']}%")
    print(f"Summary JSON saved to                  : {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_full_benchmark()
