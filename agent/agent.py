import os
import sys
import json
import pathlib
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Making sure sibling folders are importable 
BASE_DIR = pathlib.Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "agent"))
sys.path.append(str(BASE_DIR / "rag"))
sys.path.append(str(BASE_DIR / "database"))

from rag import search_contract
from database import get_patient, get_ticket
from extractor import extract_referral_from_pdf, extract_bill_from_pdf
from checklist import generate_checklist

# Load environment 
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)


# Tool 1: search_contract_tool 
def search_contract_tool(query: str) -> str:
    """
    Searches the legal contract using RAG and returns the most relevant clauses.

    This is the same search_contract() from rag.py — wrapped here as a tool
    so the Gemini agent can call it when it needs to look up a contract rule.

    Args:
        query: plain English question about the contract

    Returns:
        A formatted string with the top matching clause texts
    """
    print(f"[agent] Tool called: search_contract_tool('{query}')")
    results = search_contract(query, top_k=2)
    if not results:
        return "No relevant contract clauses found."

    output = []
    for r in results:
        output.append(f"[{r['clause']}]: {r['text']}")
    return "\n\n".join(output)


# Tool 2: get_checklist_item 
def get_checklist_item(item_number: int, checklist_results: list) -> str:
    """
    Fetches the result of a specific checklist item by number.

    Args:
        item_number:       1-6, the checklist item to fetch
        checklist_results: the list of checklist results from generate_checklist()

    Returns:
        A formatted string with the item's status, reason, and clause
    """
    print(f"[agent] Tool called: get_checklist_item({item_number})")
    for item in checklist_results:
        if item["item"] == item_number:
            status = item["status"]
            icon   = "✅ PASS" if status == "PASS" else "❌ FAIL"
            return (
                f"Checklist Item {item_number}: {item['title']}\n"
                f"Status: {icon}\n"
                f"Reason: {item['reason']}\n"
                f"Contract Clause: {item['clause']}"
            )
    return f"Checklist item {item_number} not found. Valid range is 1-6."


# Tool 3: get_patient_info 
def get_patient_info_tool(ticket_id: int) -> str:
    """
    Fetches patient and ticket details from the SQLite database.

    Args:
        ticket_id: the active audit ticket ID

    Returns:
        A formatted string with patient and ticket details
    """
    print(f"[agent] Tool called: get_patient_info_tool({ticket_id})")
    patient_record = get_patient(ticket_id)
    ticket_record  = get_ticket(ticket_id)

    if not patient_record:
        return f"No patient found for ticket ID {ticket_id}"

    return (
        f"Patient Name: {patient_record.get('patient_name')}\n"
        f"Date of Birth: {patient_record.get('dob')}\n"
        f"Gender: {patient_record.get('gender')}\n"
        f"Test Ordered: {patient_record.get('test_type')}\n"
        f"Hospital: {patient_record.get('hospital_name')}\n"
        f"Insurance: {patient_record.get('insurance_info')}\n"
        f"Allergies: {patient_record.get('allergies')}\n"
        f"Ticket Status: {ticket_record.get('status') if ticket_record else 'Unknown'}\n"
        f"Ticket Created: {ticket_record.get('created_at') if ticket_record else 'Unknown'}"
    )


# Tool definitions for Gemini function calling 
TOOL_DEFINITIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_contract_tool",
                description=(
                    "Search the legal contract for relevant clauses. "
                    "Use this when the auditor asks about contract rules, "
                    "billing compliance, turnaround times, or any legal requirement."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Plain English question about the contract"
                        )
                    },
                    required=["query"]
                )
            ),
            types.FunctionDeclaration(
                name="get_checklist_item",
                description=(
                    "Get the result of a specific checklist item by number (1-6). "
                    "Use this when the auditor asks why an item passed or failed, "
                    "or wants details about a specific checklist result."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "item_number": types.Schema(
                            type=types.Type.INTEGER,
                            description="The checklist item number between 1 and 6"
                        )
                    },
                    required=["item_number"]
                )
            ),
            types.FunctionDeclaration(
                name="get_patient_info",
                description=(
                    "Get patient and ticket details from the database. "
                    "Use this when the auditor asks about the patient's name, "
                    "test ordered, insurance, or ticket status."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "ticket_id": types.Schema(
                            type=types.Type.INTEGER,
                            description="The active audit ticket ID"
                        )
                    },
                    required=["ticket_id"]
                )
            )
        ]
    )
]


def run_chatbot(
    user_message: str,
    conversation_history: list,
    ticket_id: int,
    checklist_results: list
) -> tuple[str, list]:
    """
    Runs one turn of the auditor chatbot.

    This is a ReAct-style agent loop (Module 7):
        1. Send user message + history + tools to Gemini
        2. If Gemini calls a tool → execute it → send result back
        3. Loop until Gemini produces a final text response
        4. Return the response and updated history

    Session memory (Module 9) is maintained by keeping the full
    conversation_history list and passing it to every Gemini call.

    LangGraph concept (Module 10): think of this as two nodes —
        "call_gemini" node → "call_tool" node → back to "call_gemini"
        until the agent produces a final response with no more tool calls.

    Args:
        user_message:         the auditor's question
        conversation_history: list of previous messages (session memory)
        ticket_id:            active audit ticket ID
        checklist_results:    list of checklist results from generate_checklist()

    Returns:
        (response_text, updated_conversation_history)
    """

    # Step 1: Add user message to conversation history
    conversation_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    # Step 2: System instruction — tells Gemini what role it plays
    system_instruction = f"""
You are BillGuard AI — an expert healthcare billing auditor assistant.
You are currently auditing ticket ID {ticket_id}.

You have access to three tools:
- search_contract_tool: search the legal contract for relevant clauses
- get_checklist_item: get the result of a specific checklist item (1-6)
- get_patient_info: get patient and ticket details from the database

When answering questions:
- Always cite the specific contract clause when relevant
- Be concise and professional
- If a checklist item failed, explain exactly why and which contract rule was violated
- Always use tools to look up information rather than guessing
"""

    # Step 3: ReAct loop — keep calling Gemini until no more tool calls
    max_iterations = 5   # safety limit to prevent infinite loops
    iteration      = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[agent] Gemini call #{iteration}...")

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation_history,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=TOOL_DEFINITIONS,
                temperature=0.0
            )
        )

        candidate = response.candidates[0]
        parts      = candidate.content.parts

        # Step 4: Check if Gemini wants to call a tool
        tool_calls = [p for p in parts if p.function_call is not None]

        if not tool_calls:
            # No tool calls — Gemini produced a final text response
            final_text = "".join(p.text for p in parts if p.text)
            conversation_history.append(
                types.Content(role="model", parts=parts)
            )
            print(f"[agent] Final response produced")
            return final_text, conversation_history

        # Step 5: Execute each tool call and collect results
        tool_results = []
        for part in parts:
            if part.function_call is None:
                continue

            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)

            print(f"[agent] Executing tool: {fn_name}({fn_args})")

            # Route to the correct Python function
            if fn_name == "search_contract_tool":
                result = search_contract_tool(fn_args.get("query", ""))

            elif fn_name == "get_checklist_item":
                result = get_checklist_item(
                    int(fn_args.get("item_number", 1)),
                    checklist_results
                )

            elif fn_name == "get_patient_info":
                result = get_patient_info_tool(
                    int(fn_args.get("ticket_id", ticket_id))
                )

            else:
                result = f"Unknown tool: {fn_name}"

            tool_results.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": result}
                    )
                )
            )

        # Step 6: Add model's tool call + tool results to history
        conversation_history.append(
            types.Content(role="model", parts=parts)
        )
        conversation_history.append(
            types.Content(role="user", parts=tool_results)
        )

    # If we hit max iterations, return a fallback message
    return "I was unable to complete the analysis. Please try again.", conversation_history

if __name__ == "__main__":

    import time

    print("=" * 60)
    print("LOADING DATA FOR WALTER SCHAEFER (Ticket ID 2)")
    print("=" * 60)

    # Load Walter Schaefer's data
    walter_referral = extract_referral_from_pdf(
        str(BASE_DIR / "inputs" / "walter_schaefer" / "referral.pdf")
    )
    time.sleep(15)

    walter_bill = extract_bill_from_pdf(
        str(BASE_DIR / "inputs" / "walter_schaefer" / "bill.pdf")
    )
    time.sleep(15)

    # Generate checklist results
    print("\n[agent] Generating checklist for Walter Schaefer...")
    walter_checklist = generate_checklist(walter_referral, walter_bill, ticket_id=2)

    # Test the chatbot with the mentor's example question 
    print("\n" + "=" * 60)
    print("AUDITOR CHATBOT — Walter Schaefer (Ticket ID 2)")
    print("=" * 60)

    conversation_history = []

    # Question 1 — exactly from mentor's spec
    print("\n🧑 Auditor: 'Why is checklist item 4 flagged red?'")
    time.sleep(15)
    response1, conversation_history = run_chatbot(
        user_message="Why is checklist item 4 flagged red?",
        conversation_history=conversation_history,
        ticket_id=2,
        checklist_results=walter_checklist
    )
    print(f"🤖 BillGuard AI: {response1}")

    # Question 2 — follow up (tests session memory)
    print("\n🧑 Auditor: 'What test was ordered for this patient?'")
    time.sleep(15)
    response2, conversation_history = run_chatbot(
        user_message="What test was ordered for this patient?",
        conversation_history=conversation_history,
        ticket_id=2,
        checklist_results=walter_checklist
    )
    print(f"🤖 BillGuard AI: {response2}")

    # Question 3 — tests contract search
    print("\n🧑 Auditor: 'What does the contract say about turnaround time?'")
    time.sleep(15)
    response3, conversation_history = run_chatbot(
        user_message="What does the contract say about turnaround time?",
        conversation_history=conversation_history,
        ticket_id=2,
        checklist_results=walter_checklist
    )
    print(f"🤖 BillGuard AI: {response3}")

    print("\n" + "=" * 60)
    print("CHATBOT TEST COMPLETE")
    print("=" * 60)