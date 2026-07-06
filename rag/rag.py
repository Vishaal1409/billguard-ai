import os
import json
import pickle
import pathlib
import numpy as np
import pymupdf4llm
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths 
BASE_DIR      = pathlib.Path(__file__).parent.parent          # project root
CONTRACT_PATH = BASE_DIR / "inputs" / "clinical_laboratory_agreement.pdf"
INDEX_PATH    = BASE_DIR / "rag"    / "contract_index.faiss"
CHUNKS_PATH   = BASE_DIR / "rag"    / "chunks.json"

# Embedding model 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Extracting and chunking the contract

def chunk_contract(pdf_path: str = None) -> list[dict]:
    """
    Reads the clinical laboratory agreement PDF and splits it into
    meaningful chunks, one per clause wherever possible.

    Why clause-level chunks?
        Each of the 10 clauses is a self-contained legal rule. Keeping them
        as separate chunks means a search for "billing rules" retrieves
        Clause 6 cleanly, not a mix of unrelated clauses.

    Returns:
        A list of dicts, each with:
        {
            "chunk_id":   int,      # 0-based index
            "clause":     str,      # "Clause 1", "Clause 2", ... or "Header"
            "text":       str,      # the actual chunk text
        }
    """
    if pdf_path is None:
        pdf_path = str(CONTRACT_PATH)

    print(f"[rag] Reading contract: {pdf_path}")

    # Step 1: Convert PDF to plain text using pymupdf4llm
    markdown_text = pymupdf4llm.to_markdown(pdf_path)

    # Step 2: Split using RecursiveCharacterTextSplitter
    # chunk_size=600  — large enough to capture a full clause in one chunk
    # chunk_overlap=60 — small overlap so context isn't lost at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "]
    )
    raw_chunks = splitter.split_text(markdown_text)

    # Step 3: Label each chunk with its clause number
    # We detect which clause a chunk belongs to by looking for the clause
    # heading patterns that appear in the contract text
    clause_keywords = {
        "1":  ["Referral Framework", "Preferred Vendor"],
        "2":  ["Specialized Laboratory Capabilities", "Testing Suites"],
        "3":  ["Turnaround Time", "TAT", "Priority Processing"],
        "4":  ["Specimen Collection", "Chain of Custody"],
        "5":  ["Electronic Medical Records", "EMR Integration", "Result Reporting"],
        "6":  ["Patient Billing", "Insurance Coordination", "Corporate Rates", "Bill to Patient"],
        "7":  ["Medical Confidentiality", "Data Protection"],
        "8":  ["Quality Assurance", "Licensing", "Clinical Audits"],
        "9":  ["Term", "Automatic Renewal", "Modifications"],
        "10": ["Indemnification", "Termination Protocol"],
    }

    chunks = []
    for i, text in enumerate(raw_chunks):
        clause_label = "General"
        for clause_num, keywords in clause_keywords.items():
            if any(kw.lower() in text.lower() for kw in keywords):
                clause_label = f"Clause {clause_num}"
                break

        chunks.append({
            "chunk_id": i,
            "clause":   clause_label,
            "text":     text.strip()
        })

    print(f"[rag] Contract split into {len(chunks)} chunks")

    # Step 4: Print a quick summary so we can verify the important clauses
    print("\n[rag] Clause coverage summary:")
    seen = set()
    for c in chunks:
        if c["clause"] not in seen:
            preview = c["text"][:80].replace("\n", " ")
            print(f"  {c['clause']:12s} → {preview}...")
            seen.add(c["clause"])

    return chunks


# Embed chunks and build FAISS index

def build_faiss_index(chunks: list[dict] = None) -> tuple:
    """
    Converts each contract chunk into a vector embedding using
    sentence-transformers, then stores all vectors in a FAISS index.

    Why FAISS?
        Your mentor's requirements.txt includes faiss-cpu specifically for
        this. FAISS (Facebook AI Similarity Search) stores vectors and lets
        you find the most similar ones to any query vector in milliseconds —
        even across thousands of chunks.

    Why all-MiniLM-L6-v2?
        It's a small, fast, high-quality embedding model — perfect for this
        use case. It converts any text into a 384-dimensional vector that
        captures the semantic meaning of the text.

    Saves:
        rag/contract_index.faiss  — the FAISS index (binary)
        rag/chunks.json           — the original chunk texts + metadata

    Returns:
        (index, chunks) — the FAISS index object and the chunks list
    """
    if chunks is None:
        chunks = chunk_contract()

    print(f"\n[rag] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 1: Extract just the text from each chunk for embedding
    texts = [c["text"] for c in chunks]

    # Step 2: Convert all chunk texts into vectors
    # show_progress_bar=True prints a progress bar while encoding
    print(f"[rag] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Step 3: Build FAISS index
    # IndexFlatL2 = exact nearest-neighbour search using L2 (Euclidean) distance
    # d = embedding dimension (384 for all-MiniLM-L6-v2)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    print(f"[rag] FAISS index built — {index.ntotal} vectors stored (dimension: {d})")

    # Step 4: Save index to disk
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[rag] Index saved to: {INDEX_PATH}")

    # Step 5: Save chunks to disk as JSON
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"[rag] Chunks saved to: {CHUNKS_PATH}")

    return index, chunks

# Search the contract

def search_contract(query: str, top_k: int = 3) -> list[dict]:
    """
    Takes a plain English question and returns the top_k most relevant
    contract chunks.

    This is what your mentor calls "AI Contract Search" in Stage 3.
    The checklist engine (Day 5) will call this for every billing check
    to find the relevant clause before asking Gemini to make a PASS/FAIL
    decision.

    Args:
        query:  plain English question e.g. "what are the billing rules?"
        top_k:  number of chunks to return (default 3)

    Returns:
        List of dicts: [{"chunk_id": int, "clause": str, "text": str, "score": float}]
    """

    # Step 1: Load FAISS index from disk (build it first if it doesn't exist)
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        print("[rag] Index not found — building now...")
        build_faiss_index()

    index  = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Step 2: Embed the query using the same model
    model        = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Step 3: Search the index for the top_k nearest chunks
    distances, indices = index.search(query_vector, top_k)

    # Step 4: Build and return the results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:       # FAISS returns -1 if fewer results than top_k
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(dist)   # lower score = more similar
        results.append(chunk)

    return results

# Running this file directly to build the index and test search

if __name__ == "__main__":

    print("=" * 60)
    print("TASK 1 — Chunk the contract")
    print("=" * 60)
    chunks = chunk_contract()

    print("\n" + "=" * 60)
    print("TASK 2 — Build FAISS index")
    print("=" * 60)
    index, chunks = build_faiss_index(chunks)

    print("\n" + "=" * 60)
    print("TASK 3 — Test contract search")
    print("=" * 60)

    test_queries = [
        "what are the billing rules?",
        "turnaround time for test results",
        "what happens if the contract is terminated?",
        "insurance and billing method",
        "confidentiality and data protection",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_contract(query, top_k=2)
        for r in results:
            print(f"   [{r['clause']}] (score: {r['score']:.2f}) {r['text'][:120]}...")

    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETE — RAG pipeline ready!")
    print("=" * 60)