import requests, pandas as pd, time

API_KEY = os.environ.get("UMLS_API_KEY")
BASE    = "https://uts-ws.nlm.nih.gov/rest"

MEDICAL_ENTITIES = [
    "hypertension","diabetes","cancer","tumor","asthma","pneumonia",
    "hepatitis","migraine","epilepsy","depression","arthritis",
    "pancreatitis","sepsis","obesity","anemia","fibrosis","carcinoma",
    "lymphoma","leukemia","melanoma","cirrhosis","cholesterol",
    "thyroid","stroke","surgery","kidney","liver","heart","lung",
    "brain","prostate","breast","pancreas","ovary","warfarin",
    "aspirin","metformin","insulin","heparin","infection","inflammation",
    "fracture","biopsy","chemotherapy","radiotherapy","dialysis",
    "transplant","angioplasty","endoscopy","laparoscopy","blood pressure",
]

results = {}
for term in MEDICAL_ENTITIES:
    try:
        r = requests.get(f"{BASE}/search/current",
            params={"string": term, "apiKey": API_KEY, "pageSize": 1}, timeout=10)
        res = r.json()["result"]["results"]
        if not res or res[0]["ui"] == "NONE":
            continue
        cui = res[0]["ui"]
        r2 = requests.get(f"{BASE}/content/current/CUI/{cui}/relations",
            params={"apiKey": API_KEY, "pageSize": 25}, timeout=10)
        for rel in r2.json().get("result", []):
            name  = rel.get("relatedIdName", "")
            label = rel.get("relationLabel", "")
            if (label in ["RO","RB"] and name and
                len(name.split()) <= 4 and
                term.lower() not in name.lower() and
                all(ord(c) < 128 for c in name)):
                results[term] = {"substitution": name, "relation": label, "cui": cui}
                print(f"  ✅ {term:<20} → {name} [{label}]")
                break
        time.sleep(0.3)
    except Exception as e:
        print(f"  ❌ {term}: {e}")

df = pd.DataFrame(results).T.reset_index()
df.columns = ["entity","substitution","relation","cui"]
df.to_csv("umls_substitutions.csv", index=False)
print(f"\nSaved {len(results)} substitutions to umls_substitutions.csv")
