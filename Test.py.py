from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# STEP 3: Define model
MODEL_ID = "google/medgemma-4b-it"

# STEP 4: Load tokenizer and model
print("Loading MedGemma 4B... (this may take a few minutes on first run)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # efficient memory usage
    device_map="auto",             # auto-selects GPU or CPU
)

# STEP 5: Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# STEP 6: Ask medical questions
def ask_medgemma(question: str, max_new_tokens: int = 512) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable medical assistant. Answer clearly and accurately."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    output = pipe(messages, max_new_tokens=max_new_tokens)
    # Extract the assistant's reply
    return output[0]["generated_text"][-1]["content"]


# ============================================================
# EXAMPLE USAGE
# ============================================================

questions = [
    "What are the common symptoms of Type 2 diabetes?",
    "What is the difference between systolic and diastolic blood pressure?",
    "What are the first-line treatments for hypertension?"
]

for q in questions:
    print(f"\n🔵 Question: {q}")
    print(f"🟢 Answer: {ask_medgemma(q)}")
    print("-" * 60)