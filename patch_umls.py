# Run this to verify substitutions load correctly
import pandas as pd

df = pd.read_csv("umls_substitutions.csv")
lookup = dict(zip(df["entity"], df["substitution"]))
print(f"Loaded {len(lookup)} UMLS substitutions")

# Test on sample questions
questions = [
    "Does high blood pressure reduce the risk of chronic low back pain?",
    "Is endothelin-1 an aggravating factor in the development of acute pancreatitis?",
    "Is leptin involved in phagocytic NADPH oxidase overactivity in obesity?",
    "Are women who are treated for hypothyroidism at risk for pregnancy complications?",
]

import re
for q in questions:
    q_lower = q.lower()
    found = next((e for e in sorted(lookup.keys(), key=len, reverse=True)
                  if e in q_lower), None)
    if found:
        attacked = re.sub(re.escape(found), lookup[found], q, count=1, flags=re.IGNORECASE)
        print(f"\n  Entity:   {found} → {lookup[found]}")
        print(f"  Original: {q}")
        print(f"  Attacked: {attacked}")
    else:
        print(f"\n  No match: {q[:60]}")
