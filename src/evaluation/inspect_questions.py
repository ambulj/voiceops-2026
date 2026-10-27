import csv
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Capstone_Project_Dataset_FTTT_2026/practice/questions.csv', encoding='utf-8') as f:
    qs = list(csv.DictReader(f))

print(f"Total questions: {len(qs)}")
for i, q in enumerate(qs):
    print(f"[{i+1:02d}] {q['id']} | Lang: {q['language']} | Ans: {q['answerable']}")
    print(f"     Q: {q['question']}")
    print(f"     Gold: {q['gold_evidence']}")
    if q['hard_negatives']:
        print(f"     HardNeg: {q['hard_negatives']}")
    print(f"     MustInc: {q['must_include']}")
    if q['must_not_include']:
        print(f"     MustNotInc: {q['must_not_include']}")
    print("-" * 50)
