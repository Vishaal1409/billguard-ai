"""
extractor.py
------------
Stage 1 of the Healthcare Auditing Engine pipeline.

Contains three functions:
1. extract_referral_from_pdf()  - reads a referral PDF and returns 9-field JSON
2. extract_referral_from_audio() - reads a WAV audio file and returns 9-field JSON
3. extract_bill_from_pdf()      - reads a bill PDF and returns 8-field JSON

All extraction is done using Google Gemini API (google-genai).
PDF text is extracted using pymupdf4llm.
Audio is loaded using librosa and sent to Gemini natively.
"""

import os
import json
import base64
import librosa
import numpy as np
import pymupdf4llm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Initialise Gemini client ──────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Extract referral fields from a PDF file
# ══════════════════════════════════════════════════════════════════════════════

def extract_referral_from_pdf(pdf_path: str) -> dict:
    """
    Reads a referral PDF and extracts exactly 9 fields using Gemini.

    Args:
        pdf_path: Full path to the referral PDF file.

    Returns:
        A dictionary with these exact keys (as per mentor's spec Section 2A):
        hospital_name, patient_name, gender, dob, address,
        phone, test_type, allergies, insurance_info
    """

    # Step 1: Convert PDF to markdown text using pymupdf4llm
    # pymupdf4llm is better than pypdf for LLMs because it preserves
    # table structure and layout which referral forms rely on heavily
    print(f"[extractor] Reading PDF: {pdf_path}")
    markdown_text = pymupdf4llm.to_markdown(pdf_path)

    # Step 2: Build the extraction prompt
    # We tell Gemini exactly what fields to find and what format to return
    prompt = f"""
You are a medical document parser. Read the referral document below and extract 
exactly these 9 fields. Return ONLY a valid JSON object with these exact keys.
Do not add any extra fields. Do not add any explanation outside the JSON.

Fields to extract:
- hospital_name: The name of the hospital or lab facility that the patient is being referred to
- patient_name: The patient's full name
- gender: Male, Female, or Other (use the Sex field in the document)
- dob: Patient's date of birth in MM/DD/YYYY format
- address: Patient's full home address
- phone: Patient's home phone number
- test_type: The laboratory test that was ordered
- allergies: Any allergies listed (use the Other/Notes field — if it mentions allergies, state them; if none found write "None listed")
- insurance_info: The primary insurance provider and policy number

Return format — exactly this JSON structure:
{{
    "hospital_name": "...",
    "patient_name": "...",
    "gender": "...",
    "dob": "...",
    "address": "...",
    "phone": "...",
    "test_type": "...",
    "allergies": "...",
    "insurance_info": "..."
}}

REFERRAL DOCUMENT:
{markdown_text}
"""

    # Step 3: Send to Gemini and request JSON response
    print(f"[extractor] Sending referral text to Gemini ({GEMINI_MODEL})...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0   # 0 = deterministic, best for structured extraction
        )
    )

    # Step 4: Parse and return the JSON
    result = json.loads(response.text)
    print(f"[extractor] Referral extraction complete for: {result.get('patient_name', 'Unknown')}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract referral fields from an audio WAV file
# ══════════════════════════════════════════════════════════════════════════════

def extract_referral_from_audio(audio_path: str) -> dict:
    """
    Reads an audio WAV referral file and extracts exactly 9 fields using Gemini.

    The audio is a phone call between a provider's office and a lab.
    Gemini natively understands audio — no manual transcription needed.

    Args:
        audio_path: Full path to the WAV audio file.

    Returns:
        Same 9-field dictionary as extract_referral_from_pdf().
    """

    # Step 1: Load audio using librosa (from mentor's requirements.txt)
    # librosa loads the WAV and gives us the raw audio data and sample rate
    print(f"[extractor] Loading audio file: {audio_path}")
    audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    # Step 2: Convert audio to bytes for Gemini
    # Gemini accepts audio as base64-encoded bytes
    # We convert the numpy array back to 16-bit PCM WAV bytes
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes  = audio_int16.tobytes()

    # Step 3: Build the extraction prompt (same 9 fields as PDF version)
    prompt = """
You are a medical document parser. Listen to this audio recording carefully.
It is a phone call between a healthcare provider's office and a laboratory,
placing a patient referral order.

Extract exactly these 9 fields from the conversation.
Return ONLY a valid JSON object with these exact keys.
Do not add any extra fields. Do not add any explanation outside the JSON.

Fields to extract:
- hospital_name: The name of the hospital, clinic, or provider facility calling
- patient_name: The patient's full name mentioned in the call
- gender: Male, Female, or Other (based on how they refer to the patient)
- dob: Patient's date of birth in MM/DD/YYYY format
- address: Patient's home address if mentioned
- phone: Patient's phone number if mentioned
- test_type: The laboratory test being ordered
- allergies: Any allergies mentioned (if none, write "None listed")
- insurance_info: Insurance provider and policy number mentioned

Return format — exactly this JSON structure:
{
    "hospital_name": "...",
    "patient_name": "...",
    "gender": "...",
    "dob": "...",
    "address": "...",
    "phone": "...",
    "test_type": "...",
    "allergies": "...",
    "insurance_info": "..."
}
"""

    # Step 4: Send audio + prompt to Gemini (multimodal — Module 6)
    print(f"[extractor] Sending audio to Gemini ({GEMINI_MODEL}) for extraction...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part(
                inline_data=types.Blob(
                    mime_type="audio/wav",
                    data=audio_bytes
                )
            ),
            types.Part(text=prompt)
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    # Step 5: Parse and return the JSON
    result = json.loads(response.text)
    print(f"[extractor] Audio extraction complete for: {result.get('patient_name', 'Unknown')}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Extract bill fields from a bill PDF file
# ══════════════════════════════════════════════════════════════════════════════

def extract_bill_from_pdf(pdf_path: str) -> dict:
    """
    Reads a lab invoice/bill PDF and extracts exactly 8 fields using Gemini.

    Args:
        pdf_path: Full path to the bill PDF file.

    Returns:
        A dictionary with these exact keys (as per mentor's spec Section 2B):
        test_type, patient_name, hospital_name, drawn_by, drawn_at,
        bill_info (nested dict), performed_by, insurance_coverage_applied
    """

    # Step 1: Convert bill PDF to markdown text
    print(f"[extractor] Reading bill PDF: {pdf_path}")
    markdown_text = pymupdf4llm.to_markdown(pdf_path)

    # Step 2: Build the extraction prompt
    # Note: bill_info is a NESTED JSON — this is Module 3 structured schema output
    prompt = f"""
You are a medical billing document parser. Read the invoice/bill below and extract 
exactly these 8 fields. Return ONLY a valid JSON object with these exact keys.
Do not add any extra fields. Do not add any explanation outside the JSON.

Fields to extract:
- test_type: The laboratory test that was actually billed
- patient_name: The name of the patient being billed
- hospital_name: The hospital or medical center where the test was performed
- drawn_by: The name of the doctor or physician who ordered the test (Ordered By field)
- drawn_at: The date the test was completed (Date Completed field)
- bill_info: A nested JSON object with exactly 3 keys:
    - base_cost: The base laboratory procedure charge as a number (no $ sign)
    - extra_fees: The additional processing and handling fees as a number (no $ sign)
    - net_total: The final net total due as a number (no $ sign)
- performed_by: The name of the laboratory technician who analyzed the sample (Analyzed By field)
- insurance_coverage_applied: Either "Yes" or "No" based on whether insurance was applied

Return format — exactly this JSON structure:
{{
    "test_type": "...",
    "patient_name": "...",
    "hospital_name": "...",
    "drawn_by": "...",
    "drawn_at": "...",
    "bill_info": {{
        "base_cost": 0.00,
        "extra_fees": 0.00,
        "net_total": 0.00
    }},
    "performed_by": "...",
    "insurance_coverage_applied": "Yes" or "No"
}}

BILL DOCUMENT:
{markdown_text}
"""

    # Step 3: Send to Gemini and request JSON response
    print(f"[extractor] Sending bill text to Gemini ({GEMINI_MODEL})...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    # Step 4: Parse and return the JSON
    result = json.loads(response.text)
    print(f"[extractor] Bill extraction complete for: {result.get('patient_name', 'Unknown')}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER — Run this file directly to test all 3 functions
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import pathlib

    # Base path — adjust if your folder structure is different
    BASE = pathlib.Path(__file__).parent.parent  # goes up from agent/ to project root

    print("\n" + "="*60)
    print("TEST 1: Referral PDF — Walter Schaefer")
    print("="*60)
    walter_referral = extract_referral_from_pdf(
        str(BASE / "inputs" / "walter_schaefer" / "referral.pdf")
    )
    print(json.dumps(walter_referral, indent=2))

    print("\n" + "="*60)
    print("TEST 2: Referral PDF — Cynthia Ford")
    print("="*60)
    cynthia_referral = extract_referral_from_pdf(
        str(BASE / "inputs" / "cynthia_ford" / "referral.pdf")
    )
    print(json.dumps(cynthia_referral, indent=2))

    print("\n" + "="*60)
    print("TEST 3: Audio Referral — WAV file")
    print("="*60)
    audio_referral = extract_referral_from_audio(
        str(BASE / "inputs" / "referral_audio.wav")
    )
    print(json.dumps(audio_referral, indent=2))

    print("\n" + "="*60)
    print("TEST 4: Bill PDF — Walter Schaefer")
    print("="*60)
    walter_bill = extract_bill_from_pdf(
        str(BASE / "inputs" / "walter_schaefer" / "bill.pdf")
    )
    print(json.dumps(walter_bill, indent=2))

    print("\n" + "="*60)
    print("TEST 5: Bill PDF — Cynthia Ford")
    print("="*60)
    cynthia_bill = extract_bill_from_pdf(
        str(BASE / "inputs" / "cynthia_ford" / "bill.pdf")
    )
    print(json.dumps(cynthia_bill, indent=2))

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)