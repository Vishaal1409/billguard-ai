# 🏥 BillGuard AI
### Healthcare Billing Auditing Engine

An AI-powered auditing pipeline that reads patient referrals (audio or PDF), extracts structured data, validates invoices against a legal contract using RAG, and surfaces a 6-point compliance checklist through a Streamlit web app.

---

## 🔍 What It Does

| Stage | Input | Output |
|-------|-------|--------|
| 1 — Referral intake | Audio (.wav) or PDF referral | 9-field patient JSON → SQLite ticket |
| 2 — Lab report | Lab report PDF | Attached to active ticket |
| 3 — Bill audit | Invoice PDF + contract | 6-point checklist (Pass/Fail) |

---

## 🧱 Architecture

```
Audio/PDF Referral
       │
       ▼
extract_referral.py  (Gemini multimodal extraction)
       │
       ▼
database.py          (SQLite — patients + tickets)
       │
       ▼
extract_bill.py      (Gemini bill field extraction)
       │              +
rag/build_index.py   (Contract → FAISS vector index)
       │
       ▼
rag/rag_check.py     (Semantic contract search)
       │
       ▼
checklist.py         (6-point verification)
       │
       ▼
ui/app.py            (Streamlit — Upload · Checklist · Chatbot)
```

---

## 📁 Project Structure

```
billguard-ai/
├── agent/
│   ├── extract_referral.py    # Audio/PDF → 9-field referral JSON
│   ├── extract_bill.py        # Invoice PDF → 8-field bill JSON
│   ├── checklist.py           # 6-point compliance checker
│   └── database.py            # SQLite ticket management
├── rag/
│   ├── build_index.py         # Contract PDF → FAISS index
│   └── rag_check.py           # Semantic contract search
├── ui/
│   └── app.py                 # Streamlit web interface
├── inputs/                    # Patient files (gitignored)
├── outputs/                   # Extracted JSONs (gitignored)
├── database/                  # SQLite DB (gitignored)
├── .env                       # API keys (gitignored)
├── requirements.txt
└── PIPELINE_PLAN.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Vishaal1409/billguard-ai.git
cd billguard-ai
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

---

## 🚀 Run

```bash
# 1. Build the RAG index (run once)
python rag/build_index.py

# 2. Launch the Streamlit app
streamlit run ui/app.py
```

---

## 🛠 Tech Stack

- **Google Gemini** — multimodal extraction (audio + PDF)
- **pymupdf4llm** — PDF to markdown
- **FAISS** — vector similarity search
- **sentence-transformers** — text embeddings
- **langchain-text-splitters** — contract chunking
- **SQLite** — patient ticket database
- **Streamlit** — web UI
- **librosa** — audio processing

---

## 📋 The 6-Point Checklist

The AI verifies each transaction against the referral, bill, database, and contract:

1. Patient identity match (referral vs bill)
2. Test type match (ordered vs billed)
3. Hospital name match
4. Insurance coverage applied correctly
5. Fee structure compliant with contract
6. Billing date within valid window

---

*Built as a Data Science (Agent) capstone project.*
