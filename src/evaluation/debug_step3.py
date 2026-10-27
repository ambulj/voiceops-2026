import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evaluation.eval_benchmark import run_step3_eval

res = run_step3_eval('Capstone_Project_Dataset_FTTT_2026/practice/conversations.csv', max_turns=96)
failed = [r for r in res['results'] if not r['passed']]
print(f"\nTotal failed turns: {len(failed)}")
for f in failed:
    print(f"[{f['conversation_id']} T{f['turn']}] User: '{f['user']}'")
    print(f"   ExpDec: {f['expected_decision']} | ActDec: {f['decision']}")
    print(f"   ExpTools: {f['expected_tools']} | ActTools: {f['tools_called']}")
    print(f"   Note: {f['note']}")
    print(f"   Reply: {f['agent_reply']}")
    print("-" * 50)
