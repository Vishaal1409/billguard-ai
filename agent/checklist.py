import time
import os
import sys
import json
import pathlib
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

# Make sure sibling folders are importable 
BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "agent"))
sys.path.append(str(BASE_DIR / "rag"))
sys.path.append(str(BASE_DIR / "database"))

from rag import search_contract
from extractor import extract_referral_from_pdf, extract_bill_from_pdf
from database import get_patient, get_ticket

# Load environment 
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)

# Logging 
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("checklist")


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(genai_errors.APIError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_gemini(prompt: str) -> str:
    """
    Single Gemini call with automatic retry on rate limit errors.
    Returns the raw text response.
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    return response.text


# CORE CHECKLIST LOGIC

def check_item(item_num: int, title: str, prompt: str, clause: str = "N/A") -> dict:
    """
    Runs a single checklist item by sending a prompt to Gemini.
    Expects Gemini to return JSON: {"status": "PASS"/"FAIL", "reason": "..."}

    Args:
        item_num: checklist item number (1-6)
        title:    short label for this item
        prompt:   the full prompt to send to Gemini
        clause:   relevant contract clause (if applicable)

    Returns:
        {item, title, status, reason, clause}
    """
    print(f"[checklist] Running item {item_num}: {title}...")
    raw = call_gemini(prompt)

    try:
        result = json.loads(raw)
        status = result.get("status", "FAIL").upper()
        reason = result.get("reason", "No reason provided")
    except json.JSONDecodeError:
        status = "FAIL"
        reason = f"Could not parse Gemini response: {raw[:100]}"

    print(f"[checklist] Item {item_num} → {status}")
    time.sleep(15)
    return {
        "item":   item_num,
        "title":  title,
        "status": status,
        "reason": reason,
        "clause": clause
    }


def generate_checklist(referral_json: dict, bill_json: dict, ticket_id: int) -> list:
    """
    Runs all 6 checklist items and returns the full verification report.

    Args:
        referral_json: 9-field dict from extract_referral_from_pdf()
        bill_json:     8-field dict from extract_bill_from_pdf()
        ticket_id:     active ticket ID from database

    Returns:
        List of 6 result dicts, each with item, title, status, reason, clause
    """
    print(f"\n[checklist] Starting 6-point verification for ticket {ticket_id}...")
    results = []

    # Item 1: Patient name match 
    results.append(check_item(
        item_num=1,
        title="Patient Name Match",
        clause="N/A — direct comparison",
        prompt=f"""
You are a billing auditor performing a verification check.

REFERRAL patient name: "{referral_json.get('patient_name')}"
BILL patient name:     "{bill_json.get('patient_name')}"

Task: Check if these two patient names refer to the same person.
Allow for minor formatting differences (e.g. middle initials, capitalisation).
If they match or are clearly the same person: PASS.
If they are clearly different people: FAIL.

Return ONLY this JSON:
{{"status": "PASS" or "FAIL", "reason": "one sentence explanation"}}
"""
    ))

    # Item 2: Test type match 
    results.append(check_item(
        item_num=2,
        title="Test Type Match",
        clause="N/A — direct comparison",
        prompt=f"""
You are a billing auditor performing a verification check.

REFERRAL test ordered: "{referral_json.get('test_type')}"
BILL test billed:      "{bill_json.get('test_type')}"

Task: Check if the test that was billed matches the test that was ordered
in the referral. Allow for minor naming differences for the same test.
If they match: PASS. If different tests: FAIL.

Return ONLY this JSON:
{{"status": "PASS" or "FAIL", "reason": "one sentence explanation"}}
"""
    ))

    # Item 3: Hospital name match 
    results.append(check_item(
        item_num=3,
        title="Hospital Name Match",
        clause="N/A — direct comparison",
        prompt=f"""
You are a billing auditor performing a verification check.

REFERRAL hospital/lab: "{referral_json.get('hospital_name')}"
BILL hospital name:    "{bill_json.get('hospital_name')}"

Task: Check if the facility named in the referral matches the facility
that issued the bill. Note: the referral may name the lab facility while
the bill names the hospital — use reasonable judgement.
If they are consistent: PASS. If clearly mismatched: FAIL.

Return ONLY this JSON:
{{"status": "PASS" or "FAIL", "reason": "one sentence explanation"}}
"""
    ))

    # Item 4: Billing method compliance (uses RAG — Clause 6) 
    billing_clauses = search_contract("billing method Bill to Patient insurance", top_k=2)
    clause_text_4   = "\n".join([c["text"] for c in billing_clauses])
    clause_ref_4    = ", ".join(list(dict.fromkeys([c["clause"] for c in billing_clauses])))

    results.append(check_item(
        item_num=4,
        title="Billing Method Compliance",
        clause=clause_ref_4,
        prompt=f"""
You are a billing auditor performing a contract compliance check.

BILL insurance coverage applied: "{bill_json.get('insurance_coverage_applied')}"
REFERRAL insurance info:         "{referral_json.get('insurance_info')}"

CONTRACT RULE:
{clause_text_4}

Task: According to the contract rule above, billing must be processed via
"Bill to Patient" or "Bill to Third-Party Payer/Insurance" only.
Check if the billing method used in this bill complies with this contract rule.
If compliant: PASS. If non-compliant: FAIL.

Return ONLY this JSON:
{{"status": "PASS" or "FAIL", "reason": "one sentence explanation citing the contract clause"}}
"""
    ))

    # Item 5: Insurance coverage correctly applied (uses RAG — Clause 6)
    insurance_clauses = search_contract("insurance coverage applied patient billing", top_k=2)
    clause_text_5     = "\n".join([c["text"] for c in insurance_clauses])
    clause_ref_5      = ", ".join(list(dict.fromkeys([c["clause"] for c in insurance_clauses])))

    results.append(check_item(
        item_num=5,
        title="Insurance Coverage Applied",
        clause=clause_ref_5,
        prompt=f"""
You are a billing auditor performing a verification check.

BILL insurance coverage applied: "{bill_json.get('insurance_coverage_applied')}"
REFERRAL insurance info:         "{referral_json.get('insurance_info')}"
BILL financial details:          "{json.dumps(bill_json.get('bill_info', {}))}"

CONTRACT RULE:
{clause_text_5}

Task: Check if insurance coverage has been correctly applied to this bill.
If the referral lists an insurance provider and the bill shows insurance
was applied: PASS.
If insurance info exists in the referral but was NOT applied in the bill: FAIL.
If no insurance info exists and none was applied: PASS.

Return ONLY this JSON:
{{"status": "PASS" or "FAIL", "reason": "one sentence explanation"}}
"""
    ))

    # Item 6: Test type contract compliance (uses RAG — Clause 2/3)
    capability_clauses = search_contract("laboratory testing capabilities equipment test menu", top_k=2)
    clause_text_6      = "\n".join([c["text"] for c in capability_clauses])
    clause_ref_6       = ", ".join(list(dict.fromkeys([c["clause"] for c in capability_clauses])))

    results.append(check_item(
        item_num=6,
        title="Test Type Contract Compliance",
        clause=clause_ref_6,
        prompt=f"""
You are a billing auditor performing a contract compliance check.

BILL test billed: "{bill_json.get('test_type')}"

CONTRACT RULE (Lab's certified testing capabilities):
{clause_text_6}

Task: Check if the test billed falls within the Laboratory's certified testing
capabilities described in the contract (routine diagnostics, endocrine assays,
toxicology panels, male fertility screens, etc.), or is a reasonably standard
clinical lab test the Laboratory would be expected to perform.
If it's within scope: PASS. If clearly outside the Lab's described capabilities: FAIL.

Return ONLY this JSON:
{{"status": "PASS" or "FAIL", "reason": "one sentence explanation citing the contract clause"}}
"""
    ))
    # Final summary 
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n[checklist] Verification complete: {passed}/6 items passed")
    return results


# TEST RUNNER
if __name__ == "__main__":

    print("=" * 60)
    print("LOADING EXTRACTED DATA FOR BOTH PATIENTS")
    print("=" * 60)

    # Extract referral and bill JSONs for both patients 
    walter_referral = extract_referral_from_pdf(
        str(BASE_DIR / "inputs" / "walter_schaefer" / "referral.pdf")
    )
    walter_bill = extract_bill_from_pdf(
        str(BASE_DIR / "inputs" / "walter_schaefer" / "bill.pdf")
    )

    cynthia_referral = extract_referral_from_pdf(
        str(BASE_DIR / "inputs" / "cynthia_ford" / "referral.pdf")
    )
    cynthia_bill = extract_bill_from_pdf(
        str(BASE_DIR / "inputs" / "cynthia_ford" / "bill.pdf")
    )

    # Walter Schaefer — Ticket ID 2
    print("\n" + "=" * 60)
    print("CHECKLIST — Walter Schaefer (Ticket ID 2)")
    print("=" * 60)
    walter_results = generate_checklist(walter_referral, walter_bill, ticket_id=2)

    print("\n── Walter Schaefer Results ──")
    for r in walter_results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"{icon} Item {r['item']}: {r['title']}")
        print(f"   Status : {r['status']}")
        print(f"   Reason : {r['reason']}")
        print(f"   Clause : {r['clause']}")

    # Cynthia Ford — Ticket ID 3 
    print("\n" + "=" * 60)
    print("CHECKLIST — Cynthia Ford (Ticket ID 3)")
    print("=" * 60)
    cynthia_results = generate_checklist(cynthia_referral, cynthia_bill, ticket_id=3)

    print("\n── Cynthia Ford Results ──")
    for r in cynthia_results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"{icon} Item {r['item']}: {r['title']}")
        print(f"   Status : {r['status']}")
        print(f"   Reason : {r['reason']}")
        print(f"   Clause : {r['clause']}")

    print("\n" + "=" * 60)
    print("ALL CHECKLIST TESTS COMPLETE")
    print("=" * 60)