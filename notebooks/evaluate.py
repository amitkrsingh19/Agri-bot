import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_service import get_retrieval_chain
from app.config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

# Load eval questions using path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
eval_dataset_path = os.path.join(script_dir, "eval_dataset.json")

print(f"Script directory: {script_dir}")
print(f"Looking for eval dataset at: {eval_dataset_path}")
print(f"File exists: {os.path.exists(eval_dataset_path)}")

if not os.path.exists(eval_dataset_path):
    # Fallback: look in current directory
    fallback_path = "notebooks/eval_dataset.json"
    if os.path.exists(fallback_path):
        eval_dataset_path = fallback_path
        print(f"Using fallback path: {eval_dataset_path}")
    else:
        print(f"Error: Cannot find eval_dataset.json")
        sys.exit(1)

with open(eval_dataset_path, 'r', encoding='utf-8-sig') as f:
    content = f.read()
    print(f"File content length: {len(content)} chars")
    if len(content) == 0:
        print("Error: File is empty")
        sys.exit(1)
    eval_data = json.loads(content)

print("Loading RAG chain...")
chain = get_retrieval_chain()

# Run each question through your actual pipeline
results = []
print(f"Evaluating {len(eval_data)} questions...")
for i, item in enumerate(eval_data, 1):
    question = item["question"]
    print(f"  [{i}/{len(eval_data)}] {question}")
    
    try:
        result = chain.invoke({"input": question})
        
        results.append({
            "question": question,
            "answer": result.get("answer", ""),
            "contexts": result.get("sources", []),
            "ground_truth": item["ground_truth"],
        })
    except Exception as e:
        print(f"    Error: {e}")
        results.append({
            "question": question,
            "answer": f"Error: {str(e)}",
            "contexts": [],
            "ground_truth": item["ground_truth"],
        })

# Save results
eval_results_path = os.path.join(script_dir, "eval_results.json")
with open(eval_results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON results to {eval_results_path}")

# Also save as CSV for easier viewing
csv_results = []
for r in results:
    csv_results.append({
        "question": r["question"],
        "answer": r["answer"][:100] if r["answer"] else "",  # Truncate for CSV
        "ground_truth": r["ground_truth"],
        "num_contexts": len(r["contexts"]),
    })

df = pd.DataFrame(csv_results)
csv_path = os.path.join(script_dir, "eval_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV summary to {csv_path}")

print("\nEvaluation complete!")
print(df.to_string())