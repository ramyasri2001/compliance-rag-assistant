import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import json
import time
import dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings

# LOAD ENV
dotenv.load_dotenv("/Users/ramyasrikanugula/Desktop/compliance-rag-assistant/.env")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
print(f"Anthropic Key loaded: {'Yes' if ANTHROPIC_API_KEY else 'No'}")
print(f"Voyage Key loaded:    {'Yes' if VOYAGE_API_KEY else 'No'}")

# TEST DATA
test_data = [
    {"question": "What defects were found in truck 7?", "ground_truth": "Low pressure on the steer axle tires was found in truck 7"},
    {"question": "Who is the driver of truck 9?", "ground_truth": "Carla Nguyen is the driver of truck 9"},
    {"question": "Which trucks have defects?", "ground_truth": "Trucks 7, 9, and 23 have defects with low tire pressure on the steer axle"},
    {"question": "What is the VIN of truck 12?", "ground_truth": "The VIN of truck 12 is 2NP2HM6X3MM234567"},
    {"question": "What company do the trucks belong to?", "ground_truth": "The trucks belong to Lone Star Freight LLC"},
    {"question": "What is the plate number of truck 7?", "ground_truth": "The plate number of truck 7 is TX-TRK-007"},
    {"question": "What route does truck 7 operate on?", "ground_truth": "Truck 7 operates on the Dallas TX to Houston TX route"},
    {"question": "When was truck 7 inspected?", "ground_truth": "Truck 7 was inspected on 06/10/2026"},
    {"question": "What is the IFTA license number for truck 12?", "ground_truth": "The IFTA license number for truck 12 is TX-IFTA-62153"},
    {"question": "Who drives truck 12?", "ground_truth": "Sandra Okafor drives truck 12"},
]

BASE_URL = "http://localhost:8000"

# UPLOAD DOCUMENTS
print("\nUploading fleet documents...")
pdfs_path = "fleet_docs/pdfs"
files_to_upload = [
    "dvir_truck7_06172026.pdf",
    "dvir_truck9_05202026.pdf",
    "dvir_truck12_05312026.pdf",
    "dvir_truck23_05282026.pdf",
    "ifta_tax_form_truck12_Q2_2026.pdf",
    "ifta_tax_form_truck9_Q2_2026.pdf",
]

for filename in files_to_upload:
    filepath = os.path.join(pdfs_path, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/upload",
                files={"file": (filename, f, "application/pdf")}
            )
        resp_data = response.json()
        print(f"Uploaded {filename} - {resp_data.get('chunks_created', '?')} chunks")
    else:
        print(f"File not found: {filepath}")

print("\nWaiting 5 seconds...")
time.sleep(5)

# ASK QUESTIONS
print("Running evaluation questions...")
questions = []
answers = []
contexts = []
ground_truths = []

for item in test_data:
    question = item["question"]
    ground_truth = item["ground_truth"]
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        data = response.json()
        answer = data.get("answer", "")
        source_type = data.get("source_type", "")
        if answer and "error" not in data:
            questions.append(question)
            answers.append(answer)
            contexts.append([answer[:500]])
            ground_truths.append(ground_truth)
            print(f"OK: {question[:50]}... [{source_type}]")
        else:
            print(f"SKIP: {question[:50]}... - {data.get('error', '')}")
    except Exception as e:
        print(f"ERROR: {question[:50]}... - {e}")

print(f"\n{len(questions)} answers collected")

# BUILD DATASET
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

# SETUP LLM AND EMBEDDINGS
llm = ChatAnthropic(
    model="claude-opus-4-6",
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=2048
)

embeddings = VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model="voyage-code-3"
)

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# RUN RAGAS
print("\nRunning RAGAS evaluation (this may take 2 hours)...")

run_config = RunConfig(
    timeout=180,
    max_retries=5,
    max_wait=90,
    max_workers=1
)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config,
)

# SAVE RESULTS SAFELY
print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
print(f"Raw result: {result}")

def safe_float(val):
    try:
        if hasattr(val, '__iter__'):
            vals = [v for v in val if v is not None and str(v) != 'nan']
            return sum(vals) / len(vals) if vals else 0.0
        return float(val)
    except:
        return 0.0

faith = safe_float(result['faithfulness'])
relevancy = safe_float(result['answer_relevancy'])
precision = safe_float(result['context_precision'])

print(f"Faithfulness:      {faith:.4f}")
print(f"Answer Relevancy:  {relevancy:.4f}")
print(f"Context Precision: {precision:.4f}")
print("="*50)

with open("ragas_results.json", "w") as f:
    json.dump({
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "num_questions": len(questions)
    }, f, indent=2)

print("Results saved to ragas_results.json")