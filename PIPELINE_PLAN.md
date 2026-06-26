# Healthcare Auditing Engine — Pipeline Plan

## Project Folder Structure
```
project/
├── inputs/                        ← All mentor test files (audio, PDFs, latex docs)
│   ├── referral_audio.wav
│   ├── clinical_laboratory_agreement.pdf
│   ├── walter_schaefer/           ← referral.tex, bill.tex
│   └── cynthia_ford/              ← referral.tex, bill.tex
├── outputs/                       ← Extracted JSON results go here
├── rag/                           ← FAISS index + contract chunks
├── agent/                         ← Core pipeline Python files
├── ui/                            ← Streamlit app
├── database/                      ← patients.db (SQLite)
└── .env                           ← GEMINI_API_KEY
```

---

## Stage 1 — Referral Upload & Extraction

### `agent/extract_referral.py`
**What it does:**
- Accepts either audio (.wav) OR PDF as input
- If audio: uses google-genai to transcribe + extract fields natively (Gemini multimodal)
- If PDF: uses pymupdf4llm to get markdown text, then sends to Gemini with JSON schema prompt
- Outputs a validated JSON with exactly these 9 fields:
  - hospital_name, patient_name, gender, dob (MM/DD/YYYY)
  - address, phone, test_type, allergies, insurance_info

### `agent/database.py`
**What it does:**
- Creates SQLite DB at `database/patients.db`
- Table: `patients` — stores all 9 referral fields + ticket_status + timestamp
- Table: `tickets` — links patient → lab_report_path → bill_path → checklist_result
- Functions: create_ticket(), get_patient(), update_ticket()

---

## Stage 2 — Lab Report Attachment

### (No separate Python file needed)
- UI uploads the lab report PDF
- File is saved to `outputs/lab_report_{patient_id}.pdf`
- Ticket record is updated in SQLite with the lab report path

---

## Stage 3 — Bill Extraction + RAG Check

### `agent/extract_bill.py`
**What it does:**
- Accepts invoice bill PDF as input
- Uses pymupdf4llm → markdown → Gemini JSON extraction
- Outputs 8 fields: test_type, patient_name, hospital_name, drawn_by,
  drawn_at, bill_info (nested: base_cost, extra_fees, net_total),
  performed_by, insurance_coverage_applied (Yes/No)

### `rag/build_index.py`
**What it does:**
- Reads `clinical_laboratory_agreement.pdf` using pymupdf4llm
- Splits into chunks using langchain-text-splitters (RecursiveCharacterTextSplitter)
- Embeds chunks using sentence-transformers (all-MiniLM-L6-v2)
- Stores FAISS index to `rag/contract_index.faiss`
- Run ONCE at startup to build the index

### `rag/rag_check.py`
**What it does:**
- Loads FAISS index from disk
- Takes a checklist question as input
- Embeds the question → searches FAISS → retrieves top-k contract chunks
- Sends chunks + question to Gemini → returns Pass/Fail + reason + clause reference

### `agent/checklist.py`
**What it does:**
- Compares referral JSON vs bill JSON vs patient DB record
- Runs 6 checks (e.g. test_type match, patient name match, insurance applied, fee compliance)
- For each check that needs contract validation → calls rag_check.py
- Returns a list of 6 items: {item, status: "pass"/"fail", reason, clause}

---

## UI

### `ui/app.py`
**What it does:**
- Streamlit single-page app with 3 sections:
  1. **Upload panel** — file uploaders for Referral (audio/PDF), Lab Report (PDF), Invoice (PDF)
  2. **Checklist sheet** — renders 6-point checklist with green ✅ / red ❌ badges
  3. **Auditor chatbot** — text input → searches docs via RAG → Gemini answers in plain English

---

## Key Libraries Per File
| File | Libraries |
|------|-----------|
| extract_referral.py | google-genai, pymupdf4llm, librosa, python-dotenv |
| database.py | sqlite3 (stdlib), python-dotenv |
| extract_bill.py | google-genai, pymupdf4llm, python-dotenv |
| build_index.py | pymupdf4llm, langchain-text-splitters, sentence-transformers, faiss-cpu |
| rag_check.py | sentence-transformers, faiss-cpu, google-genai |
| checklist.py | (calls extract + rag modules) |
| ui/app.py | streamlit, pillow |

---

## Coding Order (recommended)
1. `database.py` — foundation everything else saves to
2. `extract_referral.py` — test with Walter Schaefer's referral
3. `extract_bill.py` — test with Walter Schaefer's bill
4. `rag/build_index.py` → `rag/rag_check.py` — build + test contract search
5. `checklist.py` — wire together all 3 sources
6. `ui/app.py` — connect everything into Streamlit
