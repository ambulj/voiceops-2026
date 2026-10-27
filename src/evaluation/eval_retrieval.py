import csv
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.rag.retriever import PrecedenceRetriever

def run_retrieval_eval():
    retriever = PrecedenceRetriever()
    
    with open('Capstone_Project_Dataset_FTTT_2026/practice/questions.csv', encoding='utf-8') as f:
        questions = list(csv.DictReader(f))

    total = len(questions)
    recall_hits = 0
    total_gold_count = 0
    exact_match_count = 0
    trap_violations = 0
    ans_correct = 0

    print(f"Evaluating {total} questions on Precedence Retriever...")
    for idx, q in enumerate(questions):
        qid = q['id']
        text = q['question']
        expected_ans = q['answerable'] == 'True'
        raw_gold = q['gold_evidence'].strip()
        gold_secs = [s.strip() for s in raw_gold.split(' ; ') if s.strip()]
        hard_negs = [s.strip() for s in q['hard_negatives'].split(' ; ') if s.strip()]

        results, is_ans = retriever.retrieve(text, top_k=4, entity_hint=q.get('entities'))
        retrieved_ids = [r.section_id for r in results]

        # Check answerability prediction
        if is_ans == expected_ans:
            ans_correct += 1

        # Check hard negative traps
        trapped = any(hn in retrieved_ids for hn in hard_negs)
        if trapped:
            trap_violations += 1

        # Check recall
        if not expected_ans:
            # For unanswerable, perfect if retrieved_ids is empty
            if len(retrieved_ids) == 0:
                exact_match_count += 1
                recall_hits += 1
            total_gold_count += 1
        else:
            hits = sum(1 for g in gold_secs if g in retrieved_ids)
            recall_hits += hits
            total_gold_count += len(gold_secs)
            if hits == len(gold_secs):
                exact_match_count += 1
            else:
                print(f"[{qid}] Missing gold: {[g for g in gold_secs if g not in retrieved_ids]} | Got: {retrieved_ids}")

    gold_recall = (recall_hits / total_gold_count) * 100 if total_gold_count else 0
    ans_acc = (ans_correct / total) * 100

    print("\n" + "=" * 50)
    print(f"Total Questions: {total}")
    print(f"Answerability Accuracy: {ans_acc:.1f}% ({ans_correct}/{total})")
    print(f"Gold Evidence Recall: {gold_recall:.1f}% ({recall_hits}/{total_gold_count})")
    print(f"Full Exact Match on Gold: {exact_match_count}/{total} ({(exact_match_count/total)*100:.1f}%)")
    print(f"Hard Negative Trap Violations: {trap_violations}")
    print("=" * 50)

if __name__ == '__main__':
    run_retrieval_eval()
