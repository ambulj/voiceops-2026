import csv
import sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

with open('Capstone_Project_Dataset_FTTT_2026/practice/conversations.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

convs = defaultdict(list)
for r in rows:
    convs[r['conversation_id']].append(r)

print(f"Total conversations: {len(convs)} | Total turns: {len(rows)}")
for cid in sorted(convs.keys())[:5]:
    c_turns = convs[cid]
    first = c_turns[0]
    print(f"\n=== {cid} ({first['scenario']}) Lang: {first['language']} Order: {first['order_id']} Cust: {first['customer_id']} ===")
    for t in c_turns:
        print(f" Turn {t['turn']}: [{t['expected_decision']}] '{t['customer_message']}'")
        if t['expected_tool_calls']:
            print(f"    Tools: {t['expected_tool_calls']}")
        if t['must_include']:
            print(f"    MustInc: {t['must_include']}")
        if t['must_not_include']:
            print(f"    MustNotInc: {t['must_not_include']}")
        print(f"    Gold: {t['gold_evidence']}")
