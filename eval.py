import time
import csv
from embed import get_embedding, cosine_similarity
from main1 import run_query as llama_query
from main2 import run_query as mistral_query

# ✅ Define your evaluation questions
questions = [
    "What is the difference between applying for leave through VTOP and applying for leave manually?",
    "What are the restrictions for weekend outings on VTOP, and what happens if a student returns late?",
    "How can a student raise a complaint about hostel maintenance issues using VTOP?",
    "Explain the steps to check your semester timetable on VTOP in simple words for a new student.",
    "If VTOP does not allow me to apply for leave due to technical issues, what alternative method is available?"
]

# ✅ Dataset reference (manual text for similarity check)
with open("dataset.txt", "r", encoding="utf-8") as f:
    dataset_text = f.read()

dataset_embedding = get_embedding(dataset_text)

# ✅ Function to evaluate model
def evaluate_model(model_name, query_func):
    results = []
    for q in questions:
        start = time.time()
        answer = query_func(q)
        end = time.time()
        duration = round(end - start, 2)

        # Embedding similarity with dataset
        ans_embedding = get_embedding(answer)
        sim = round(cosine_similarity(dataset_embedding, ans_embedding), 3)

        # Precision proxy → keyword overlap
        precision = len(set(q.lower().split()) & set(answer.lower().split())) / len(set(q.lower().split()))

        results.append({
            "Question": q,
            "Model": model_name,
            "Answer": answer,
            "Response Time (s)": duration,
            "Similarity": sim,
            "Precision": round(precision, 3)
        })
    return results

# ✅ Run both models
llama_results = evaluate_model("LLaMA", llama_query)
mistral_results = evaluate_model("Mistral", mistral_query)

all_results = llama_results + mistral_results

# ✅ Save to CSV
with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Question", "Model", "Answer", "Response Time (s)", "Similarity", "Precision"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

print("✅ Evaluation complete! Results saved to evaluation_results.csv")
