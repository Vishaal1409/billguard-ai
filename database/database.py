"""
database.py
------------
Stage 1 (continued) and Stage 2 of the Healthcare Auditing Engine pipeline.

Handles all SQLite database operations:
- Creating the schema (patients, tickets, documents tables)
- Validating/inserting patients
- Creating and updating active tickets
- Attaching lab report / bill documents to a ticket

This is the persistence layer that everything else (checklist, agent, UI)
will read from and write to.
"""

import sqlite3
import os
import pathlib
from datetime import datetime

# ── Database file location ──────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.parent   # project root
DB_PATH  = BASE_DIR / "database" / "audit.db"


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    """
    Opens a connection to the SQLite database file.
    Creates the file automatically if it doesn't exist yet.

    row_factory = sqlite3.Row lets us access columns by name
    (e.g. row['patient_name']) instead of by index (row[1]) — much safer
    and more readable.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Create the schema (3 tables)
# ══════════════════════════════════════════════════════════════════════════════

def create_tables():
    """
    Creates all 3 tables if they don't already exist:

    1. patients   — one row per unique patient, holds all 9 referral fields
    2. tickets    — one row per audit case, links to a patient, tracks status
    3. documents  — one row per uploaded file, links to a ticket

    Safe to call this every time the app starts — CREATE TABLE IF NOT EXISTS
    means it won't wipe existing data or throw an error if tables already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── Table 1: patients ───────────────────────────────────────────────
    # Stores the exact 9 fields from mentor's spec Section 2A
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            hospital_name   TEXT,
            patient_name    TEXT NOT NULL,
            gender          TEXT,
            dob             TEXT NOT NULL,
            address         TEXT,
            phone           TEXT,
            test_type       TEXT,
            allergies       TEXT,
            insurance_info  TEXT,
            created_at      TEXT NOT NULL
        )
    """)

    # ── Table 2: tickets ────────────────────────────────────────────────
    # One ticket per audit case. status moves: ACTIVE -> LAB_ATTACHED -> AUDITED
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER NOT NULL,
            status          TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    """)

    # ── Table 3: documents ──────────────────────────────────────────────
    # Links uploaded files (lab report, bill) to a ticket
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id       INTEGER NOT NULL,
            doc_type        TEXT NOT NULL,
            file_path       TEXT NOT NULL,
            uploaded_at     TEXT NOT NULL,
            FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[database] Tables created/verified at: {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Validate patient, create ticket, attach documents
# ══════════════════════════════════════════════════════════════════════════════

def validate_patient(referral_json: dict) -> int:
    """
    Checks if a patient already exists in the database (matched by
    patient_name + dob — this pair is unique enough for our test data).

    - If found: returns the existing patient_id (no duplicate row created)
    - If not found: inserts a new patient row using all 9 referral fields
      and returns the new patient_id

    Args:
        referral_json: dict with the 9 fields from extractor.py
                        (hospital_name, patient_name, gender, dob, address,
                         phone, test_type, allergies, insurance_info)

    Returns:
        patient_id (int)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Step 1: Check if this patient already exists (same name + dob)
    cursor.execute(
        "SELECT patient_id FROM patients WHERE patient_name = ? AND dob = ?",
        (referral_json["patient_name"], referral_json["dob"])
    )
    existing = cursor.fetchone()

    if existing:
        patient_id = existing["patient_id"]
        print(f"[database] Patient already exists: {referral_json['patient_name']} (ID {patient_id})")
        conn.close()
        return patient_id

    # Step 2: Not found — insert a new patient row
    cursor.execute("""
        INSERT INTO patients (
            hospital_name, patient_name, gender, dob, address,
            phone, test_type, allergies, insurance_info, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        referral_json.get("hospital_name"),
        referral_json.get("patient_name"),
        referral_json.get("gender"),
        referral_json.get("dob"),
        referral_json.get("address"),
        referral_json.get("phone"),
        referral_json.get("test_type"),
        referral_json.get("allergies"),
        referral_json.get("insurance_info"),
        datetime.now().isoformat()
    ))

    patient_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"[database] New patient inserted: {referral_json['patient_name']} (ID {patient_id})")
    return patient_id


def create_ticket(patient_id: int) -> int:
    """
    Creates a new active audit ticket for a given patient.

    This is Step 1's final action in the mentor's pipeline:
    "Validate in Database → Create Active Ticket"

    Args:
        patient_id: the patient_id returned by validate_patient()

    Returns:
        ticket_id (int)
    """
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO tickets (patient_id, status, created_at, updated_at)
        VALUES (?, 'ACTIVE', ?, ?)
    """, (patient_id, now, now))

    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"[database] New ticket created: ID {ticket_id} for patient {patient_id} (status: ACTIVE)")
    return ticket_id


def attach_document(ticket_id: int, doc_path: str, doc_type: str) -> int:
    """
    Links an uploaded document (lab report or bill) to an existing ticket.

    This is Stage 2 of the mentor's pipeline:
    "Upload Lab Report PDF → Attach to Active Ticket"

    Args:
        ticket_id: the ticket this document belongs to
        doc_path:  file path of the uploaded document
        doc_type:  one of "referral", "lab_report", or "bill"

    Returns:
        document_id (int)
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (ticket_id, doc_type, file_path, uploaded_at)
        VALUES (?, ?, ?, ?)
    """, (ticket_id, doc_type, doc_path, datetime.now().isoformat()))

    document_id = cursor.lastrowid

    # Also bump the ticket's updated_at and status, so we can track progress
    cursor.execute("""
        UPDATE tickets SET updated_at = ?, status = ?
        WHERE ticket_id = ?
    """, (datetime.now().isoformat(), f"{doc_type.upper()}_ATTACHED", ticket_id))

    conn.commit()
    conn.close()
    print(f"[database] Document attached: {doc_type} -> ticket {ticket_id} (doc ID {document_id})")
    return document_id


def get_patient(patient_id: int) -> dict:
    """
    Helper: fetches a patient row by patient_id and returns it as a dict.
    Useful later for the RAG checklist and chatbot (Module 9 state lookup).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_ticket(ticket_id: int) -> dict:
    """
    Helper: fetches a ticket row by ticket_id and returns it as a dict.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER — run this file directly to set up the database
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    create_tables()
    print("[database] Task 1 complete — schema is ready.\n")

    # ── Quick Task 2 test with fake data (real test happens in Task 3) ──
    print("="*60)
    print("TASK 2 QUICK TEST — using sample data")
    print("="*60)

    sample_referral = {
        "hospital_name": "Test Hospital",
        "patient_name": "Test Patient",
        "gender": "Other",
        "dob": "01/01/2000",
        "address": "123 Test St",
        "phone": "000-000-0000",
        "test_type": "Test Panel",
        "allergies": "None listed",
        "insurance_info": "Test Insurance - Policy: TEST123"
    }

    pid = validate_patient(sample_referral)
    tid = create_ticket(pid)
    attach_document(tid, "inputs/test/lab_report.pdf", "lab_report")

    print("\nFetched patient record:")
    print(get_patient(pid))

    print("\nFetched ticket record:")
    print(get_ticket(tid))

    print("\n[database] Task 2 complete — all functions working.")
