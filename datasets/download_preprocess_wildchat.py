from datasets import load_dataset
from tqdm import tqdm
import json

# Load the WildChat dataset
dataset = load_dataset("allenai/WildChat", split="train")

# Filter and collect only 10k valid entries
filtered = []
for row in tqdm(dataset, desc="Filtering dataset"):
    if row.get("language") == "English" and len(row.get("conversation", [])) >= 2:
        question = row["conversation"][0]["content"]
        answer = row["conversation"][1]["content"]

        # Word count checks
        if len(question.split()) < 100 and len(answer.split()) > 100 and len(answer.split()) < 300:
            filtered.append({
                "model": row["model"],
                "question": question,
                "answer": answer
            })

        if len(filtered) >= 10000:
            break

# Save to JSON
with open("wildchat_10k_filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

# Print basic stats
print(f"Saved {len(filtered)} entries to wildchat_10k_filtered.json")
print(f"Unique models: {len(set(entry['model'] for entry in filtered))}")
print(f"Average question length: {sum(len(e['question']) for e in filtered) / len(filtered):.2f}")
print(f"Average answer length: {sum(len(e['answer']) for e in filtered) / len(filtered):.2f}")