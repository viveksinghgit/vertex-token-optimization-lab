
import os
import json
import time
import hashlib
import pickle
import re
import json
import hashlib
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")
INDEX_FILE = Path("results/chunk_index.pkl")
FAST_MODEL = os.getenv("FAST_MODEL", "gemini-3.5-flash-lite")
SMART_MODEL = os.getenv("SMART_MODEL", "gemini-3.8-flash")
# ============================================================
# MODEL PRICING — USD per 1 million tokens
# Vertex AI Global pricing
# ============================================================

MODEL_PRICING = {
    "gemini-3.5-flash-lite": {
        "input": 0.15,
        "output": 1.25
    },
    "gemini-3.8-flash": {
        "input": 0.375,
        "output": 1.875
    }
}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
RESPONSE_CACHE_FILE = Path("results/response_cache.json")
BENCHMARK_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What are the backup frequency and retention requirements for Tier 1 systems?",
        "expected_sources": ["backup_policy.txt"],
    },
    {
        "id": "Q2",
        "question": "What are the RTO and RPO requirements for Tier 1 systems during disaster recovery?",
        "expected_sources": ["disaster_recovery.txt"],
    },
    {
        "id": "Q3",
        "question": "What security requirements must workloads follow when deployed on AKS?",
        "expected_sources": ["kubernetes_policy.txt", "security_policy.txt"],
    },
    {
    "id": "Q4",
    "question": "Design a multi-region disaster recovery architecture for AKS and analyze the tradeoffs between cost, resilience, RTO, RPO, and operational complexity.",
    "expected_sources": [
        "disaster_recovery.txt",
        "kubernetes_policy.txt",
        "security_policy.txt"
    ],
    },
    
]
EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL", "gemini-embedding-001"
)

DATA_DIR = Path("data")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

client = genai.Client(
    enterprise=True,
    project=PROJECT,
    location=LOCATION
)

SYSTEM = """
You are an enterprise cloud policy assistant.
Answer using only the supplied company policies.
Do not invent policy details.
If evidence is insufficient, say so.
Cite the document name where possible.
Keep answers concise.
"""
def calculate_cost(result):

    model = result.get("model_used", MODEL)

    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        raise ValueError(f"Pricing not configured for model: {model}")

    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)

    input_cost = (
        input_tokens / 1_000_000
    ) * pricing["input"]

    output_cost = (
        output_tokens / 1_000_000
    ) * pricing["output"]

    total_cost = input_cost + output_cost

    result["input_cost_usd"] = round(input_cost, 8)
    result["output_cost_usd"] = round(output_cost, 8)
    result["total_cost_usd"] = round(total_cost, 8)

    return result
    
def load_documents():
    documents = []

    for file in DATA_DIR.glob("*.txt"):
        documents.append({
            "name": file.name,
            "text": file.read_text(encoding="utf-8")
        })

    if not documents:
        raise RuntimeError("No policy documents found in data/")

    return documents


def embed_text(text, task_type):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )
    return np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )


def retrieve_context(question, documents, top_k=3):
    query_vector = embed_text(
        question, "RETRIEVAL_QUERY"
    )

    scored = []

    for doc in documents:
        doc_vector = embed_text(
            doc["text"], "RETRIEVAL_DOCUMENT"
        )

        similarity = float(
            np.dot(query_vector, doc_vector)
            / (
                np.linalg.norm(query_vector)
                * np.linalg.norm(doc_vector)
            )
        )

        scored.append((similarity, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "name": doc["name"],
            "text": doc["text"],
            "score": score
        }
        for score, doc in scored[:top_k]
    ]




def call_gemini(prompt, system=SYSTEM, cache=None, model=None):

    start = time.perf_counter()

    # Select model
    selected_model = model or MODEL

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW
        )
    )

    if cache:
        config.cached_content = cache

    print(f"🤖 Model selected: {selected_model}")

    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
        config=config
    )

    latency = time.perf_counter() - start
    usage = response.usage_metadata

    candidate = (
        response.candidates[0]
        if response.candidates else None
    )

    result = {
    "answer": response.text or "",
    "model_used": selected_model,
    "input_tokens": usage.prompt_token_count or 0,
    "output_tokens": usage.candidates_token_count or 0,
    "cached_tokens": getattr(
        usage, "cached_content_token_count", 0
    ) or 0,
    "thinking_tokens": getattr(
        usage, "thoughts_token_count", 0
    ) or 0,
    "total_tokens": usage.total_token_count or 0,
    "latency": round(latency, 3),
    "finish_reason": (
        str(candidate.finish_reason)
        if candidate else "NO_CANDIDATE"
    )
    }

    return calculate_cost(result)


def run_baseline(question, documents):
    full_context = "\n\n".join(
        f"DOCUMENT: {doc['name']}\n{doc['text']}"
        for doc in documents
    )

    prompt = f"""
COMPANY KNOWLEDGE BASE:
{full_context}

EMPLOYEE QUESTION:
{question}
"""

    return call_gemini(prompt)

def route_question(question):
    complex_keywords = [
        "design",
        "architecture",
        "architect",
        "compare",
        "comparison",
        "tradeoff",
        "trade-off",
        "analyze",
        "analysis",
        "recommend",
        "strategy",
        "strategies",
        "multi-region",
        "migration",
        "evaluate",
        "why",
        "impact",
    ]

    question_lower = question.lower()

    if any(keyword in question_lower for keyword in complex_keywords):
        return SMART_MODEL

    if len(question.split()) > 25:
        return SMART_MODEL

    return FAST_MODEL

def run_optimized(question, documents, chunk_index=None):
    if chunk_index is None:
        chunk_index = get_chunk_index(documents)

    relevant_chunks = retrieve_chunks(
        question,
        chunk_index,
        top_k=4
    )

    context = "\n\n".join(
        f"SOURCE: {chunk['name']}\n"
        f"CONTENT: {chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
RELEVANT POLICY EXCERPTS:

{context}

QUESTION:
{question}

Answer using only the relevant policy excerpts.
Cite the source filename.
"""

    selected_model = route_question(question)

    print(f"🧭 Routing question to: {selected_model}")

    result = call_gemini(
    prompt,
    model=selected_model
    )

    result["retrieved_chunks"] = [
        {
            "name": chunk["name"],
            "similarity": round(chunk["score"], 3),
            "text": chunk["text"]
        }
        for chunk in relevant_chunks
    ]

    return result

def load_response_cache():
    if RESPONSE_CACHE_FILE.exists():
        with open(RESPONSE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def run_optimized_cached(question, documents, chunk_index):
    cache = load_response_cache()
    print(">>> RUNNING CACHED FUNCTION <<<")
    # Create a deterministic cache key
    cache_key = hashlib.sha256(
        question.strip().lower().encode("utf-8")
    ).hexdigest()

    # Cache hit
    if cache_key in cache:
        print("\n⚡ RESPONSE CACHE HIT")

        cached_result = cache[cache_key].copy()

        cached_result["cache_hit"] = True
        cached_result["input_tokens"] = 0
        cached_result["output_tokens"] = 0
        cached_result["cached_tokens"] = 0
        cached_result["total_tokens"] = 0
        cached_result["latency"] = 0.0
        cached_result["input_cost_usd"] = 0
        cached_result["output_cost_usd"] = 0
        cached_result["total_cost_usd"] = 0

        return cached_result

    # Cache miss - call Gemini normally
    result = run_optimized(question, documents, chunk_index)

    result["cache_hit"] = False

    # Save the original result
    cache[cache_key] = result

    RESPONSE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESPONSE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    return result

def create_context_cache(documents):
    # Stable policy material is reusable across questions.
    cache_context = "\n\n".join(
        f"DOCUMENT: {doc['name']}\n{doc['text']}"
        for doc in documents
    )

    count = client.models.count_tokens(
        model=MODEL,
        contents=cache_context
    )

    print("Cache context tokens:", count.total_tokens)

    # Gemini 2.5 Flash explicit cache minimum:
    if count.total_tokens < 2048:
        raise ValueError(
            "Cache context is below the 2048-token "
            "minimum. Add more real policy content."
        )

    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=cache_context
                        )
                    ]
                )
            ],
            system_instruction=SYSTEM,
            display_name="conference-policy-cache",
            ttl="3600s"
        )
    )

    print("Created cache:", cache.name)
    return cache.name


def run_cached(question, cache_name):
    return call_gemini(
        prompt=question,
        cache=cache_name,
        system=SYSTEM
    )


def show_result(name, result):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Answer:", result["answer"])
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])
    print("Cached tokens:", result["cached_tokens"])
    print("Total tokens:", result["total_tokens"])
    print("Latency:", result["latency"], "seconds")
    print("Finish reason:", result["finish_reason"])
    print("Thinking tokens:", result["thinking_tokens"])

    if "retrieved_documents" in result:
        print("\nRetrieved documents:")
        print(result["retrieved_documents"])

    if "retrieved_chunks" in result:
        print("\nRetrieved chunks:")

        for chunk in result["retrieved_chunks"]:
            print(
                f"\nSource: {chunk['name']} | "
                f"Similarity: {chunk['similarity']}"
            )

            print(chunk["text"])


def run_benchmark(documents, chunk_index):
    benchmark_results = []

    for test in BENCHMARK_QUESTIONS:
        question = test["question"]

        print("\n" + "#" * 70)
        print(f"TEST {test['id']}: {question}")
        print("#" * 70)

        # Baseline: full documents
        baseline = run_baseline(question, documents)

        # Optimized: chunk retrieval + response cache
        chunk_result = run_optimized_cached(
            question,
            documents,
            chunk_index
        )

        benchmark_results.append({
            "id": test["id"],
            "question": question,
            "baseline": baseline,
            "chunk_retrieval": chunk_result,
            "expected_sources": test["expected_sources"],
        })

    # ============================================================
    # AGGREGATE METRICS
    # ============================================================

    baseline_tokens = sum(
        item["baseline"].get("total_tokens", 0)
        for item in benchmark_results
    )

    optimized_tokens = sum(
        item["chunk_retrieval"].get("total_tokens", 0)
        for item in benchmark_results
    )

    baseline_cost = sum(
        item["baseline"].get("total_cost_usd", 0)
        for item in benchmark_results
    )

    optimized_cost = sum(
        item["chunk_retrieval"].get("total_cost_usd", 0)
        for item in benchmark_results
    )

    baseline_input_tokens = sum(
        item["baseline"].get("input_tokens", 0)
        for item in benchmark_results
    )

    optimized_input_tokens = sum(
        item["chunk_retrieval"].get("input_tokens", 0)
        for item in benchmark_results
    )

    cache_hits = sum(
        1 for item in benchmark_results
        if item["chunk_retrieval"].get("cache_hit", False)
    )

    token_savings_pct = (
        (baseline_tokens - optimized_tokens) / baseline_tokens * 100
        if baseline_tokens else 0
    )

    input_savings_pct = (
        (baseline_input_tokens - optimized_input_tokens)
        / baseline_input_tokens * 100
        if baseline_input_tokens else 0
    )

    cost_savings_pct = (
        (baseline_cost - optimized_cost) / baseline_cost * 100
        if baseline_cost else 0
    )

    # ============================================================
    # BENCHMARK REPORT
    # ============================================================

    print("\n" + "=" * 90)
    print("OPTIMIZATION SUMMARY")
    print("=" * 90)

    print(f"{'Metric':<30}{'Baseline':>20}{'Optimized':>20}{'Savings':>18}")
    print("-" * 90)

    print(
        f"{'Total tokens':<30}"
        f"{baseline_tokens:>20,}"
        f"{optimized_tokens:>20,}"
        f"{token_savings_pct:>17.1f}%"
    )

    print(
        f"{'Input tokens':<30}"
        f"{baseline_input_tokens:>20,}"
        f"{optimized_input_tokens:>20,}"
        f"{input_savings_pct:>17.1f}%"
    )

    print(
        f"{'Estimated cost (USD)':<30}"
        f"${baseline_cost:>19.6f}"
        f"${optimized_cost:>19.6f}"
        f"{cost_savings_pct:>17.1f}%"
    )

    print("-" * 90)

    print(f"Response cache hits: {cache_hits}/{len(benchmark_results)}")
    print(f"Total estimated savings: ${baseline_cost - optimized_cost:.6f}")

    # ============================================================
    # COST PROJECTIONS
    # ============================================================

    query_count = len(benchmark_results)

    if query_count > 0:

        baseline_cost_per_query = baseline_cost / query_count
        optimized_cost_per_query = optimized_cost / query_count

        print("\nPROJECTED COST AT SCALE")
        print("-" * 60)
        print(
            f"{'Requests':<20}"
            f"{'Baseline':>18}"
            f"{'Optimized':>18}"
            f"{'Savings':>18}"
        )

        for requests in [1000, 10000, 100000]:

            projected_baseline = baseline_cost_per_query * requests
            projected_optimized = optimized_cost_per_query * requests
            projected_savings = (
                projected_baseline - projected_optimized
            )

            print(
                f"{requests:<20,}"
                f"${projected_baseline:>17.4f}"
                f"${projected_optimized:>17.4f}"
                f"${projected_savings:>17.4f}"
            )

    print("=" * 90)

    return benchmark_results

def print_benchmark_summary(results):
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    print(
        f"{'Test':<8}"
        f"{'Method':<22}"
        f"{'Input':<12}"
        f"{'Output':<12}"
        f"{'Total':<12}"
        f"{'Latency':<12}"
    )

    print("-" * 90)

    for test in results:
        for method, result in [
            ("Baseline", test["baseline"]),
            ("Chunk Retrieval", test["chunk_retrieval"]),
        ]:
            print(
                f"{test['id']:<8}"
                f"{method:<22}"
                f"{result['input_tokens']:<12}"
                f"{result['output_tokens']:<12}"
                f"{result['total_tokens']:<12}"
                f"{result['latency']:<12.2f}"
            )

    print("=" * 90)

def chunk_document(document, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = document["text"]

    # Split around headings and paragraphs first
    sections = re.split(
        r"\n(?=\d+\.\s+|\n[A-Z][A-Z\s&-]{4,}\n)",
        text
    )

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        words = section.split()

        if len(words) <= chunk_size:
            chunks.append({
                "name": document["name"],
                "text": section
            })
            continue

        # Split larger sections into overlapping chunks
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "name": document["name"],
                "text": chunk_text
            })

            if end == len(words):
                break

            start = end - overlap

    return chunks
    
def get_chunk_index(documents):
    # Detect changes to source documents
    source_text = "\n".join(
        doc["name"] + doc["text"]
        for doc in documents
    )

    source_hash = hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()

    # Reuse index if source documents haven't changed
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, "rb") as f:
                saved_index = pickle.load(f)

            if saved_index.get("source_hash") == source_hash:
                print("\nLoading persistent chunk index...")
                return saved_index["chunks"]

        except Exception as e:
            print("Index could not be loaded:", e)

    print("\nCreating chunk index and embeddings...")

    all_chunks = []

    for document in documents:
        document_chunks = chunk_document(document)

        for chunk in document_chunks:
            all_chunks.append(chunk)

    # Generate each embedding once
    for i, chunk in enumerate(all_chunks, start=1):
        print(
            f"Embedding chunk {i}/{len(all_chunks)}: "
            f"{chunk['name']}"
        )

        chunk["embedding"] = embed_text(
            chunk["text"],
            "RETRIEVAL_DOCUMENT"
        )

    # Save embeddings locally
    saved_index = {
        "source_hash": source_hash,
        "chunks": all_chunks
    }

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(saved_index, f)

    print(f"\nSaved persistent index: {INDEX_FILE}")

    return all_chunks

def retrieve_chunks(question, chunks, top_k=4):
    query_vector = embed_text(
        question,
        "RETRIEVAL_QUERY"
    )

    scored = []

    for chunk in chunks:
        doc_vector = chunk["embedding"]

        similarity = float(
            np.dot(query_vector, doc_vector)
            / (
                np.linalg.norm(query_vector)
                * np.linalg.norm(doc_vector)
            )
        )

        scored.append((similarity, chunk))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        {
            "name": chunk["name"],
            "text": chunk["text"],
            "score": score
        }
        for score, chunk in scored[:top_k]
    ]

if __name__ == "__main__":
    documents = load_documents()

    chunk_index = get_chunk_index(documents)

    results = run_benchmark(documents, chunk_index)

    print_benchmark_summary(results)

    with open("results/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nBenchmark saved to results/benchmark_results.json")
