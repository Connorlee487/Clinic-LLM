import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch
import pandas as pd

# Login to HuggingFace
login(token=os.environ.get("HF_TOKEN"))

# STEP 3: Define model
MODEL_ID = "google/medgemma-4b-it"

# STEP 4: Load tokenizer
print("Loading MedGemma 4B with 4-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 4-bit quantization — reduces memory from ~18GB to ~5GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
)

print("Model loaded! GPU memory used:", torch.cuda.memory_allocated() // 1024**2, "MB")

# STEP 5: Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# STEP 6: Ask medical questions
def ask_medgemma(question: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant. Answer clearly and accurately."},
        {"role": "user",   "content": question}
    ]
    output = pipe(messages, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"][-1]["content"]

# STEP 7: Questions
questions = [
    "Pap smears with glandular cell abnormalities: Are they detected by rapid prescreening?",
    "Are patients with Werlhof's disease at increased risk for bleeding complications when undergoing cardiac surgery?",
    "Is there a connection between sublingual varices and hypertension?",
    "Does health information exchange reduce redundant imaging?",
    "The promise of specialty pharmaceuticals: are they worth the price?",
    "The colour of pain: can patients use colour to describe osteoarthritis pain?",
    "Is fluoroscopy essential for retrieval of lower ureteric stones?",
    "Increased neutrophil migratory activity after major trauma: a factor in the etiology of acute respiratory distress syndrome?",
    "Does the SCL 90-R obsessive-compulsive dimension identify cognitive impairments?",
    "Can transcranial direct current stimulation be useful in differentiating unresponsive wakefulness syndrome from minimally conscious state patients?",
    "Does prostate morphology affect outcomes after holmium laser enucleation?",
    "Injury and poisoning mortality among young men--are there any common factors amenable to prevention?",
    "Are stroke patients' reports of home blood pressure readings reliable?",
    "Is endothelin-1 an aggravating factor in the development of acute pancreatitis?",
    "Is leptin involved in phagocytic NADPH oxidase overactivity in obesity?",
    "Are women who are treated for hypothyroidism at risk for pregnancy complications?",
    "Does high blood pressure reduce the risk of chronic low back pain?",
    "Do the changes in the serum levels of IL-2, IL-4, TNFalpha, and IL-6 reflect the inflammatory activity in the patients with post-ERCP pancreatitis?",
    "Ovarian torsion in children: is oophorectomy necessary?",
    "Transsphenoidal pituitary surgery in Cushing's disease: can we predict outcome?"
]

results = []
for i, q in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] {q[:60]}...")
    ans = ask_medgemma(q)
    print(f"Answer: {ans[:100]}...")
    results.append({"question": q, "medgemma_answer": ans})

# Save to Excel
df = pd.DataFrame(results)
df.to_excel("medgemma_answers_delta.xlsx", index=False)
df.to_csv("medgemma_answers_delta.csv", index=False)
print(f"\n✅ Saved {len(results)} answers to medgemma_answers_delta.xlsx")