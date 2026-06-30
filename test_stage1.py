"""
test_stage1.py
---------------
Day 3, Task 3 — Tests the FULL Stage 1 pipeline end to end, exactly as
described in mentor's spec:

    Referral Uploaded (Audio/PDF) -> Extract fields -> Validate in Database
    -> Create Active Ticket

This chains together:
    extractor.py  (Day 2 — extract_referral_from_pdf)
    database.py   (Day 3 — validate_patient, create_ticket)

Run this from the project root:
    python test_stage1.py
"""

import sys
import pathlib

# Make sure Python can find agent/ and database/ as importable folders
sys.path.append(str(pathlib.Path(__file__).parent / "agent"))
sys.path.append(str(pathlib.Path(__file__).parent / "database"))

from extractor import extract_referral_from_pdf
from database import validate_patient, create_ticket, get_patient, get_ticket

BASE = pathlib.Path(__file__).parent


def run_stage1(patient_folder: str):
    """
    Runs the complete Stage 1 flow for one patient folder.

    Args:
        patient_folder: e.g. "walter_schaefer" or "cynthia_ford"

    Returns:
        (patient_id, ticket_id)
    """
    print("\n" + "="*60)
    print(f"STAGE 1 PIPELINE — {patient_folder}")
    print("="*60)

    referral_pdf_path = str(BASE / "inputs" / patient_folder / "referral.pdf")

    # Step 1: Extract fields from the referral PDF (Day 2 — extractor.py)
    print("\n[Step 1] Extracting referral fields...")
    referral_json = extract_referral_from_pdf(referral_pdf_path)

    # Step 2: Validate in database — insert patient if new (Day 3 — database.py)
    print("\n[Step 2] Validating patient in database...")
    patient_id = validate_patient(referral_json)

    # Step 3: Create active ticket (Day 3 — database.py)
    print("\n[Step 3] Creating active ticket...")
    ticket_id = create_ticket(patient_id)

    # Step 4: Confirm everything by reading it back from the database
    print("\n[Step 4] Confirming records in database...")
    patient_record = get_patient(patient_id)
    ticket_record = get_ticket(ticket_id)

    print(f"\n✅ Patient ID:  {patient_id}")
    print(f"✅ Ticket ID:   {ticket_id}")
    print(f"✅ Patient record: {patient_record}")
    print(f"✅ Ticket record:  {ticket_record}")

    return patient_id, ticket_id


if __name__ == "__main__":

    # ── Run for Walter Schaefer ─────────────────────────────────────────
    walter_patient_id, walter_ticket_id = run_stage1("walter_schaefer")

    # ── Run for Cynthia Ford ────────────────────────────────────────────
    cynthia_patient_id, cynthia_ticket_id = run_stage1("cynthia_ford")

    # ── Final summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — FINAL SUMMARY")
    print("="*60)
    print(f"Walter Schaefer  -> Patient ID: {walter_patient_id}, Ticket ID: {walter_ticket_id}")
    print(f"Cynthia Ford     -> Patient ID: {cynthia_patient_id}, Ticket ID: {cynthia_ticket_id}")
    print("\n[test_stage1] Stage 1 pipeline test complete — both patients processed successfully.")