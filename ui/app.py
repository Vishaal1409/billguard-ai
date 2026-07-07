import os
import sys
import json
import time
import tempfile
import pathlib
import streamlit as st

# Make sure sibling folders are importable 
BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "agent"))
sys.path.append(str(BASE_DIR / "rag"))
sys.path.append(str(BASE_DIR / "database"))

from extractor import extract_referral_from_pdf, extract_referral_from_audio, extract_bill_from_pdf
from database import create_tables, validate_patient, create_ticket, attach_document
from checklist import generate_checklist
from agent import run_chatbot

# PAGE CONFIG & THEME
st.set_page_config(
    page_title="BillGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F0F4F8;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A365D;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Header banner */
    .main-header {
        background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        color: white;
    }

    /* Section headers */
    .section-header {
        background: #2B6CB0;
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Upload cards */
    .upload-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #2B6CB0;
    }

    /* Checklist items */
    .checklist-pass {
        background: #F0FFF4;
        border: 1px solid #9AE6B4;
        border-left: 5px solid #38A169;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .checklist-fail {
        background: #FFF5F5;
        border: 1px solid #FEB2B2;
        border-left: 5px solid #E53E3E;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .checklist-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }
    .checklist-reason {
        font-size: 0.9rem;
        color: #4A5568;
    }
    .checklist-clause {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.25rem;
        font-style: italic;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Chat bubbles */
    .chat-user {
        background: #2B6CB0;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .chat-bot {
        background: white;
        color: #1A365D;
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .chat-clear {
        clear: both;
    }

    /* Status badge */
    .badge-pass {
        background: #38A169;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-fail {
        background: #E53E3E;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 2px solid #E2E8F0;
        margin: 2rem 0;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALISATION

def init_session_state():
    """Initialise all session state variables on first load."""
    defaults = {
        "referral_json":       None,
        "bill_json":           None,
        "ticket_id":           None,
        "patient_id":          None,
        "checklist_results":   None,
        "chat_history":        [],       # displayed messages {role, content}
        "agent_history":       [],       # Gemini Content objects for memory
        "stage":               1,        # 1=referral, 2=lab report, 3=bill, 4=done
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Ensure DB tables exist 
create_tables()



# SIDEBAR

with st.sidebar:
    st.markdown("## 🏥 BillGuard AI")
    st.markdown("**Healthcare Billing Auditing Engine**")
    st.markdown("---")

    st.markdown("### 📋 Pipeline Stages")
    stage = st.session_state.get("stage", 1)

    stages = [
        ("1️⃣", "Referral Upload",   stage > 1),
        ("2️⃣", "Lab Report",        stage > 2),
        ("3️⃣", "Bill Upload",       stage > 3),
        ("4️⃣", "Audit Complete",    stage > 4),
    ]
    for icon, label, done in stages:
        if done:
            st.markdown(f"✅ ~~{icon} {label}~~")
        else:
            st.markdown(f"{icon} {label}")

    st.markdown("---")

    # Patient info if loaded
    if st.session_state.referral_json:
        ref = st.session_state.referral_json
        st.markdown("### 🧑‍⚕️ Active Patient")
        st.markdown(f"**{ref.get('patient_name', 'Unknown')}**")
        st.markdown(f"DOB: {ref.get('dob', '—')}")
        st.markdown(f"Test: {ref.get('test_type', '—')}")
        st.markdown(f"Ticket ID: {st.session_state.ticket_id or '—'}")

    st.markdown("---")

    # Reset button
    if st.button("🔄 Start New Audit", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("*Built for Data Science (Agent) Internship 2026*")


# MAIN HEADER

st.markdown("""
<div class="main-header">
    <h1>🏥 BillGuard AI</h1>
    <p>Healthcare Billing Auditing Engine — AI-powered compliance verification</p>
</div>
""", unsafe_allow_html=True)


# PANEL 1 — UPLOAD PANEL

st.markdown('<div class="section-header">📤 Panel 1 — Document Upload</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Upload 1: Referral 
with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Patient Referral")
    st.markdown("Upload the referral letter (PDF) or audio call (WAV)")

    referral_file = st.file_uploader(
        "Referral file",
        type=["pdf", "wav"],
        key="referral_uploader",
        label_visibility="collapsed"
    )

    if referral_file and st.session_state.referral_json is None:
        with st.spinner("🔍 Extracting referral fields..."):
            # Save to temp file
            suffix = ".pdf" if referral_file.type == "application/pdf" else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(referral_file.read())
                tmp_path = tmp.name

            try:
                if suffix == ".pdf":
                    result = extract_referral_from_pdf(tmp_path)
                else:
                    result = extract_referral_from_audio(tmp_path)

                # Save to DB
                patient_id = validate_patient(result)
                ticket_id  = create_ticket(patient_id)

                st.session_state.referral_json = result
                st.session_state.patient_id    = patient_id
                st.session_state.ticket_id     = ticket_id
                st.session_state.stage         = 2

                st.success(f"✅ Referral extracted! Ticket #{ticket_id} created.")

            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.referral_json:
        st.markdown("**Extracted fields:**")
        st.json(st.session_state.referral_json)

    st.markdown('</div>', unsafe_allow_html=True)

# Upload 2: Lab Report 
with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 🧪 Lab Report")
    st.markdown("Upload the lab results PDF after the test is completed")

    lab_file = st.file_uploader(
        "Lab report file",
        type=["pdf"],
        key="lab_uploader",
        label_visibility="collapsed",
        disabled=st.session_state.ticket_id is None
    )

    if lab_file and st.session_state.ticket_id:
        with st.spinner("📎 Attaching lab report to ticket..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(lab_file.read())
                tmp_path = tmp.name

            try:
                attach_document(
                    st.session_state.ticket_id,
                    tmp_path,
                    "lab_report"
                )
                st.session_state.stage = 3
                st.success(f"✅ Lab report attached to Ticket #{st.session_state.ticket_id}")
            except Exception as e:
                st.error(f"❌ Attachment failed: {str(e)}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.ticket_id is None:
        st.info("⏳ Upload referral first to enable this step.")

    st.markdown('</div>', unsafe_allow_html=True)

# Upload 3: Invoice Bill
with col3:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 💰 Invoice Bill")
    st.markdown("Upload the final billing invoice PDF from the lab")

    bill_file = st.file_uploader(
        "Invoice bill file",
        type=["pdf"],
        key="bill_uploader",
        label_visibility="collapsed",
        disabled=st.session_state.stage < 3
    )

    if bill_file and st.session_state.bill_json is None and st.session_state.stage >= 3:
        with st.spinner("🔍 Extracting bill fields..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(bill_file.read())
                tmp_path = tmp.name

            try:
                result = extract_bill_from_pdf(tmp_path)
                attach_document(
                    st.session_state.ticket_id,
                    tmp_path,
                    "bill"
                )
                st.session_state.bill_json = result
                st.success("✅ Bill extracted successfully!")

            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.bill_json:
        st.markdown("**Extracted fields:**")
        st.json(st.session_state.bill_json)

    if st.session_state.stage < 3:
        st.info("⏳ Upload lab report first to enable this step.")

    st.markdown('</div>', unsafe_allow_html=True)


# PANEL 2 — CHECKLIST DASHBOARD

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">✅ Panel 2 — 6-Point Verification Checklist</div>', unsafe_allow_html=True)

if st.session_state.referral_json and st.session_state.bill_json:

    # Run checklist button
    if st.session_state.checklist_results is None:
        if st.button("🚀 Run Compliance Checklist", type="primary", use_container_width=True):
            with st.spinner("🤖 Running AI verification across all 6 checklist items... (this takes ~2 minutes)"):
                try:
                    results = generate_checklist(
                        st.session_state.referral_json,
                        st.session_state.bill_json,
                        st.session_state.ticket_id
                    )
                    st.session_state.checklist_results = results
                    st.session_state.stage = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Checklist failed: {str(e)}")

    # Display checklist results
    if st.session_state.checklist_results:
        results = st.session_state.checklist_results

        # Summary metrics
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = len(results) - passed

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;font-weight:800;color:#38A169">{passed}/6</div>
                <div style="color:#4A5568">Items Passed</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;font-weight:800;color:#E53E3E">{failed}/6</div>
                <div style="color:#4A5568">Items Failed</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            verdict = "COMPLIANT" if failed == 0 else "REVIEW REQUIRED"
            color   = "#38A169" if failed == 0 else "#D69E2E"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.4rem;font-weight:800;color:{color}">{verdict}</div>
                <div style="color:#4A5568">Overall Verdict</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Individual checklist items
        for r in results:
            is_pass   = r["status"] == "PASS"
            css_class = "checklist-pass" if is_pass else "checklist-fail"
            icon      = "✅" if is_pass else "❌"
            badge     = f'<span class="badge-pass">PASS</span>' if is_pass else f'<span class="badge-fail">FAIL</span>'

            st.markdown(f"""
            <div class="{css_class}">
                <div class="checklist-title">{icon} Item {r['item']}: {r['title']} {badge}</div>
                <div class="checklist-reason">{r['reason']}</div>
                <div class="checklist-clause">📜 {r['clause']}</div>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.referral_json is None:
    st.info("⏳ Upload a patient referral in Panel 1 to begin the audit.")
elif st.session_state.bill_json is None:
    st.info("⏳ Upload the invoice bill in Panel 1 to run the checklist.")


# PANEL 3 — AUDITOR CHATBOT

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">💬 Panel 3 — Auditor Chatbot</div>', unsafe_allow_html=True)

if st.session_state.checklist_results is None:
    st.info("⏳ Run the compliance checklist first to enable the chatbot.")
else:
    st.markdown("Ask any question about this audit — the AI will search the contract and checklist results to answer.")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the audit... e.g. 'Why is item 4 flagged red?'"):

        # Add user message to display history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("🤖 BillGuard AI is thinking..."):
                try:
                    response, updated_history = run_chatbot(
                        user_message=prompt,
                        conversation_history=st.session_state.agent_history,
                        ticket_id=st.session_state.ticket_id,
                        checklist_results=st.session_state.checklist_results
                    )
                    st.session_state.agent_history = updated_history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.write(response)

                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.error(error_msg)

    # Suggested questions
    if len(st.session_state.chat_history) == 0:
        st.markdown("**💡 Suggested questions:**")
        suggestions = [
            "Why is checklist item 4 flagged red?",
            "What test was ordered for this patient?",
            "What does the contract say about billing?",
            "Is the insurance coverage correct?",
            "What does the contract say about turnaround time?"
        ]
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": suggestion
                    })
                    with st.spinner("🤖 BillGuard AI is thinking..."):
                        try:
                            response, updated_history = run_chatbot(
                                user_message=suggestion,
                                conversation_history=st.session_state.agent_history,
                                ticket_id=st.session_state.ticket_id,
                                checklist_results=st.session_state.checklist_results
                            )
                            st.session_state.agent_history = updated_history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")