"""
Minimal test to verify the LEADS model can generate output at all.
Uses a very short input to isolate whether the model works on MPS.
"""
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "zifeng-ai/leads-mistral-7b-v1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)
model.eval()
print("✓ Model loaded.\n")

# --- Test 1: Extremely short prompt ---
short_prompt = "Extract the drug name from this text: Patients received 100mg of adalimumab. Return as JSON."

print(f"=== Test 1: Short prompt ({len(short_prompt)} chars) ===")
messages = [{"role": "user", "content": short_prompt}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)

print(f"  Input token count: {input_ids.shape[-1]}")
print(f"  Starting generation (max 128 tokens)...")

start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - start

generated_tokens = output_ids[0][input_ids.shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"  Generation took: {elapsed:.1f}s")
print(f"  Generated {len(generated_tokens)} tokens")
print(f"  Response: {response}\n")

# --- Test 2: Medium prompt (closer to real usage but still small) ---
medium_prompt = (
    "You are a clinical data extraction expert. "
    "Extract the following fields: trial_name, dosage, method. "
    "If a field is not found, use \"N/A\".\n\n"
    "Research Paper Text:\n"
    "In a Phase III randomized controlled trial (ATLAS study), "
    "patients with rheumatoid arthritis received adalimumab 40mg "
    "subcutaneously every two weeks for 24 weeks.\n\n"
    "Return your answer as a single JSON object."
)

print(f"=== Test 2: Medium prompt ({len(medium_prompt)} chars) ===")
messages = [{"role": "user", "content": medium_prompt}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)

print(f"  Input token count: {input_ids.shape[-1]}")
print(f"  Starting generation (max 256 tokens)...")

start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - start

generated_tokens = output_ids[0][input_ids.shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"  Generation took: {elapsed:.1f}s")
print(f"  Generated {len(generated_tokens)} tokens")
print(f"  Response: {response}\n")

print("=== Tests complete ===")
