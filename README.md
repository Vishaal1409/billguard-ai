# 🏥 BillGuard AI
### Healthcare Billing Auditing Engine

An AI-powered auditing pipeline that reads patient referrals (audio or PDF), extracts structured data, validates invoices against a legal contract using RAG, and surfaces a 6-point compliance checklist through a Streamlit web app.

> Built as a Data Science (Agent) capstone project — Internship 2026

---

## 📌 Project Overview

**Who uses this app?** A Billing Auditor who manages clinic billing.

**Two parties are involved in every transaction:**
- **The Hospital** — sends the patient referral
- **The Lab** — runs the test and sends the bill

**What the app does:** It reads a patient referral (audio or PDF), matches the patient against a database, reads the final lab invoice, and uses an AI-powered contract search system (RAG) to verify the entire transaction against a legal agreement — producing a 6-point PASS/FAIL checklist.

---

## 🔍 Pipeline Overview

```
Stage 1 — Referral Intake
    Audio (.wav) or PDF referral
           │
           ▼
    agent/extractor.py          ← Gemini multimodal extraction
    (extract_referral_from_pdf / extract_referral_from_audio)
           │
           ▼ 9-field referral JSON
           │
    database/database.py        ← SQLite persistence
    (validate_patient → create_ticket)
           │
           ▼ Active ticket created

Stage 2 — Lab Report Attachment
    Lab Report PDF uploaded
           │
           ▼
    database/database.py
    (attach_document → ticket updated)

Stage 3 — Bill Audit
    Invoice PDF uploaded
           │
           ▼
    agent/extractor.py          ← Gemini bill extraction
    (extract_bill_from_pdf)
           │
           ▼ 8-field bill JSON
           │              +
    rag/rag.py                  ← Contract RAG search
    (search_contract → top clauses)
           │
           ▼
    agent/checklist.py          ← 6-point verification
    (generate_checklist → PASS/FAIL per item)
           │
           ▼
    ui/app.py                   ← Streamlit web interface
    (Upload Panel · Checklist Dashboard · Auditor Chatbot)
```

---

## 📁 Project Structure

```
billguard-ai/
│
├── agent/
│   ├── extractor.py        # All extraction logic (referral PDF, audio, bill PDF)
│   ├── checklist.py        # 6-point compliance verification engine
│   └── agent.py            # Auditor chatbot with tool calling + session memory
│
├── rag/
│   ├── rag.py              # Contract chunking, FAISS index, contract search
│   └── chunks.json         # Pre-chunked contract clauses (auto-generated)
│
├── database/
│   └── database.py         # SQLite schema + patient/ticket/document functions
│
├── ui/
│   └── app.py              # Streamlit single-page web application
│
├── inputs/                 # Patient test files — gitignored (PHI data)
│   ├── walter_schaefer/
│   │   ├── referral.pdf
│   │   └── bill.pdf
│   ├── cynthia_ford/
│   │   ├── referral.pdf
│   │   └── bill.pdf
│   ├── referral_audio.wav
│   └── clinical_laboratory_agreement.pdf
│
├── outputs/                # Extracted JSONs — gitignored
├── database/               # SQLite audit.db — gitignored
├── test_stage1.py          # End-to-end Stage 1 pipeline test
├── .env                    # API keys — gitignored
├── .env.example            # Template for environment variables
├── requirements.txt        # All dependencies
└── PIPELINE_PLAN.md        # Detailed pipeline design notes
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vishaal1409/billguard-ai.git
cd billguard-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your values:
```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

> **Note:** The mentor's spec originally specified `gemini-1.5-flash` but this model returned a 404 on the current API version. `gemini-2.5-flash` is the current working replacement in the same Gemini Flash family and produces identical results.

### 4. Add your test input files
Place these files in the `inputs/` folder (not included in repo — PHI data):
```
inputs/
├── walter_schaefer/referral.pdf
├── walter_schaefer/bill.pdf
├── cynthia_ford/referral.pdf
├── cynthia_ford/bill.pdf
├── referral_audio.wav
└── clinical_laboratory_agreement.pdf
```

### 5. Build the RAG index (run once)
```bash
python rag/rag.py
```
This reads the contract PDF, chunks it into 16 clause-level segments, embeds them using `sentence-transformers`, and saves the FAISS index to `rag/contract_index.faiss`.

---

## 🚀 Running the Project

### Run the full Streamlit app
```bash
streamlit run ui/app.py
```

### Test individual pipeline stages
```bash
# Test Stage 1 — referral extraction + database
python test_stage1.py

# Test extraction only
python agent/extractor.py

# Test RAG contract search
python rag/rag.py

# Test 6-point checklist
python agent/checklist.py
```

---

## 📋 Data Formats

### A. Referral JSON (9 fields — from audio or PDF)

| Field | Type | Description |
|-------|------|-------------|
| `hospital_name` | Text | The hospital sending the patient |
| `patient_name` | Text | Patient's full name |
| `gender` | Text | Male, Female, or Other |
| `dob` | Text (MM/DD/YYYY) | Patient's date of birth |
| `address` | Text | Patient's home address |
| `phone` | Text | Patient's contact number |
| `test_type` | Text | The laboratory test requested |
| `allergies` | Text | Any listed patient allergies |
| `insurance_info` | Text | Insurance provider and policy number |

### B. Bill JSON (8 fields — from invoice PDF)

| Field | Type | Description |
|-------|------|-------------|
| `test_type` | Text | The test that was actually billed |
| `patient_name` | Text | Name of the billed patient |
| `hospital_name` | Text | Facility where the test was performed |
| `drawn_by` | Text | The doctor who ordered the test |
| `drawn_at` | Text | The date the test was completed |
| `bill_info` | Nested JSON | `{base_cost, extra_fees, net_total}` |
| `performed_by` | Text | The lab technician who analyzed the sample |
| `insurance_coverage_applied` | Yes/No | Whether insurance covered any costs |

---

## 🗄️ Database Schema

Three tables stored in `database/audit.db`:

### `patients` table
Stores all 9 referral fields plus a unique `patient_id` and `created_at` timestamp. One row per unique patient (matched by `patient_name` + `dob`).

### `tickets` table
One audit ticket per case. Tracks `status` which progresses through:
`ACTIVE` → `LAB_REPORT_ATTACHED` → `BILL_ATTACHED`

### `documents` table
Links uploaded files (lab report, bill) to their ticket via `ticket_id`, `doc_type`, and `file_path`.

---

## 📜 The Contract — RAG Source (10 Clauses)

The `clinical_laboratory_agreement.pdf` is the legal agreement between **Aura Reproductive Health & Fertility Clinic** and **Nexus Diagnostics & Bio-Analytics Corp.** It contains 10 clauses that the RAG system indexes and searches:

| Clause | Title | Key Rule |
|--------|-------|----------|
| 1 | Referral Framework | Lab is the preferred vendor; orders go through EHR or authorized requisitions |
| 2 | Lab Capabilities | Lab must maintain certified equipment for all tests ordered |
| 3 | Turnaround Time | Hormone assays: 4-6 hours. Routine diagnostics: 24 hours max |
| 4 | Specimen Handling | Lab must document chain of custody for all specimens |
| 5 | EMR Integration | Results must be pushed to provider dashboard immediately after verification |
| 6 | Billing | Must be "Bill to Patient" or "Bill to Third-Party/Insurance". No other method allowed |
| 7 | Confidentiality | All patient data must comply with privacy regulations |
| 8 | Quality Assurance | Lab must hold all required licenses and allow annual audits |
| 9 | Term & Renewal | Agreement lasts 12 months, auto-renews unless 60-day written notice given |
| 10 | Termination | Either party can terminate with 60-day notice. Immediate termination for license suspension or breach |

---

## ✅ The 6-Point Verification Checklist

The AI compares the referral, bill, database, and contract to generate this checklist:

| Item | What It Checks | Sources Used |
|------|---------------|--------------|
| 1 | Patient name matches between referral and bill | referral JSON vs bill JSON |
| 2 | Test type ordered matches test type billed | referral JSON vs bill JSON |
| 3 | Hospital name matches between referral and bill | referral JSON vs bill JSON |
| 4 | Billing method complies with contract | bill JSON vs Clause 6 (RAG) |
| 5 | Insurance coverage correctly applied | bill JSON vs Clause 6 (RAG) |
| 6 | Ordering doctor consistent across documents | referral JSON vs bill JSON |

Each item returns:
```json
{
  "item": 1,
  "title": "Patient Name Match",
  "status": "PASS",
  "reason": "Patient names are identical across referral and bill.",
  "clause": "N/A — direct comparison"
}
```

---

## 🤖 The Auditor Chatbot

The chatbot (built in `agent/agent.py`) uses Gemini function calling with three tools:

| Tool | What it does |
|------|-------------|
| `search_contract_tool(query)` | Searches the contract via RAG and returns relevant clauses |
| `get_checklist_item(item_number)` | Fetches why a specific checklist item passed or failed |
| `get_patient_info(ticket_id)` | Retrieves patient and ticket details from SQLite |

Session memory keeps conversation history so the auditor can ask follow-up questions:

```
Auditor: "Why is checklist item 4 flagged red?"
AI: "Item 4 failed because the billing method used does not comply
     with Clause 6 of the agreement, which requires billing to be
     processed via 'Bill to Patient' or 'Bill to Third-Party/Insurance' only."

Auditor: "Which clause covers this?"
AI: "That would be Clause 6 — Patient Billing, Insurance Coordination
     & Corporate Rates."
```

---

## 🛠️ Tech Stack

| Library | Role in Project |
|---------|----------------|
| `google-genai` | Calls Gemini API for JSON extraction, checklist, and chatbot |
| `pymupdf4llm` | Converts PDFs to clean markdown text for LLM processing |
| `pypdf` | PDF file reading |
| `faiss-cpu` | Vector similarity search for contract RAG |
| `sentence-transformers` | Converts contract chunks to embeddings (all-MiniLM-L6-v2) |
| `langchain-text-splitters` | Splits contract into clause-level chunks |
| `librosa` | Loads and processes the audio referral WAV file |
| `python-dotenv` | Loads API keys from .env file |
| `pillow` | Image handling for PDF rendering |
| `streamlit` | Single-page web UI |
| `tenacity` | Automatic retry logic for Gemini API rate limit errors |

---

## 🧪 Test Cases

Two complete patient test cases provided by mentor:

### Walter Schaefer (Patient ID: 2, Ticket ID: 2)
- **Test ordered:** Complete Blood Count (CBC) with Differential
- **Insurance:** Cigna
- **Checklist result:** 5/6 PASS

### Cynthia Ford (Patient ID: 3, Ticket ID: 3)
- **Test ordered:** Comprehensive Metabolic Panel (CMP)
- **Insurance:** Aetna
- **Checklist result:** 5/6 PASS

### Audio referral
- WAV file of a phone call placing a lab referral order
- Extracted using Gemini's native audio understanding (multimodal)

---

## 📝 Key Implementation Notes

**Why `gemini-2.5-flash` instead of `gemini-1.5-flash`:**
The mentor's spec specified `gemini-1.5-flash` but this model returned a 404 NOT_FOUND error on the current free tier API. `gemini-2.5-flash` is the current working model in the same Gemini Flash family. This change is documented in `.env.example`.

**Why all extraction is in one file (`extractor.py`):**
All three extraction functions share the same Gemini client setup, retry logic, and `.env` config. Keeping them in one file avoids duplicate setup code and keeps Stage 1 self-contained.

**Why `tenacity` is used:**
`gemini-2.5-flash` has a free tier limit of 5 requests per minute. Tenacity automatically retries failed calls with exponential backoff (2s → 4s → 8s) instead of crashing immediately on rate limit errors.

**Why patient data is gitignored:**
All files in `inputs/` contain Protected Health Information (PHI). These are never committed to GitHub — only the code that processes them is version controlled.

---

## 📅 Development Timeline

| Day | Date | What was built |
|-----|------|---------------|
| 1 | Jun 26 | Project setup, folder structure, .env, dependencies |
| 2 | Jun 29| `extractor.py` — PDF + audio extraction → JSON |
| 3 | Jun 30 | `database.py` — SQLite schema + patient/ticket functions |
| 4 | Jul 1 | `rag.py` — contract chunking + FAISS index + search |
| 5 | Jul 2 | `checklist.py` — 6-point verification engine |
| 6 | Jul 3 | `agent.py` — chatbot with tool calling + session memory |
| 7 | Jul 7 | `ui/app.py` — Streamlit web interface |
| 8 | Jul 8 | Testing, cleanup, final spec check |

---

*Built by Vishaal — Data Science (Agent) Internship Capstone, 2026*
