import os
import operator
from typing import Annotated, TypedDict, List, Dict, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# --- Configuration & Mock RAG ---

KNOWLEDGE_BASE_PATH = "data/knowledge_base.md"

def load_knowledge_base():
    """Loads the knowledge base content."""
    try:
        with open(KNOWLEDGE_BASE_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: Knowledge base not found."

# Simple mock retrieval: In a real scenario, use vector DB.
# Here we return the full content because it's small (~20 lines).
def retrieve_docs(query: str) -> str:
    kb_content = load_knowledge_base()
    return kb_content

# --- Mock Tool ---

import csv
from datetime import datetime

def mock_lead_capture(name: str, email: str, platform: str):
    """Mocks sending lead data to a backend and saves to CSV."""
    print(f"\n[SYSTEM] Lead captured successfully: {name}, {email}, {platform}\n")
    
    # Save to file
    try:
        file_exists = os.path.isfile('leads.csv')
        with open('leads.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'name', 'email', 'platform'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'name': name,
                'email': email,
                'platform': platform
            })
    except PermissionError:
        print("[ERROR] Could not write to leads.csv. Is the file open?")
        return "Lead captured (but could not save to CSV - file is likely open)."
    except Exception as e:
        print(f"[ERROR] CSV Write failed: {e}")
        return "Lead captured (Database error)."
    
    return "Lead captured successfully."

# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: Optional[str]
    lead_info: Dict[str, Optional[str]] # name, email, platform

# --- Nodes ---

# Using Local Llama 3 via Ollama
# Ensure you have ollama installed and run: `ollama pull llama3`
llm = ChatOllama(model="llama3", temperature=0)

def detect_intent(state: AgentState):
    """Analyzes the last user message to determine intent."""
    messages = state['messages']
    last_user_msg = messages[-1].content
    
    # Get the last bot message if it exists
    last_bot_msg = "None"
    if len(messages) > 1 and messages[-2].type == "ai":
        last_bot_msg = messages[-2].content

    system_prompt = (
        "You are an intent classifier for AutoStream, a video editing SaaS. "
        "Classify the user's intent into exactly one of these categories: "
        "'GREETING', 'PRODUCT_INQUIRY', 'HIGH_INTENT', 'PROVIDING_INFO'.\n"
        "\n"
        "Definitions:\n"
        "- GREETING: Casual hellos, polite meaningless pleasantries (e.g. 'hi', 'hello').\n"
        "- PRODUCT_INQUIRY: Asking about pricing, features, policies. INCLUDES 'I would like to see/review/know about...'.\n"
        "- HIGH_INTENT: Explicitly stating desire to buy, sign up, try, or use the product. Phrases like 'I want to try', 'Sign me up', 'Interested in Pro plan', 'I would like the basic plan', 'I will go with the pro plan'. EXCLUDES 'I would like to see/review'. INCLUDES indirect choices like 'plan seems like a good fit', 'sounds perfect', 'I will take it'.\n"
        "- PROVIDING_INFO: The user is providing specific details (name, email, platform) in response to the bot's question. INCLUDES short answers like 'youtube', 'john', 'yes'.\n"
        "\n"
        "IMPORTANT: If the Last Bot Message was asking a question (e.g. 'I need your name', 'which platform'), and the User Message is a direct answer, you MUST classify as PROVIDING_INFO.\n"
        "\n"
        f"Context:\nLast Bot Message: \"{last_bot_msg}\"\n"
        f"User Message: \"{last_user_msg}\"\n"
        "\n"
        "Output ONLY the category name."
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt)])
    intent = response.content.strip().upper()
    
    # Fallback/Normalization
    valid_intents = {'GREETING', 'PRODUCT_INQUIRY', 'HIGH_INTENT', 'PROVIDING_INFO'}
    # Helper to find if the intent word is IN the response (Llama 3 can be chatty)
    found_intent = None
    for v in valid_intents:
        if v in intent:
            found_intent = v
            break
            
    if found_intent:
        intent = found_intent
    else:
        intent = 'PRODUCT_INQUIRY' # Default fallback
        
    return {"intent": intent}

def handle_greeting(state: AgentState):
    return {"messages": [AIMessage(content="Hi there! I'm the AutoStream assistant. I can help you with pricing, features, or getting started. How can I help?")]}

def handle_inquiry(state: AgentState):
    """RAG-based response."""
    messages = state['messages']
    last_user_msg = messages[-1].content
    
    knowledge = retrieve_docs(last_user_msg)
    
    system_prompt = (
        "You are a helpful assistant for AutoStream. Answer the user's question "
        "using ONLY the following context. If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{knowledge}"
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_user_msg)])
    return {"messages": [response]}

def handle_lead_qualification(state: AgentState):
    """Manages the dialogue to collect lead info."""
    messages = state['messages']
    last_user_msg = messages[-1].content
    lead_info = state.get('lead_info', {"name": None, "email": None, "platform": None})
    
    extraction_prompt = (
        "Extract the following fields from the conversation history if available: "
        "Name, Email, Creator Platform (YouTube, Instagram, etc.). "
        "Return as JSON: {\"name\": ..., \"email\": ..., \"platform\": ...}. "
        "If a field is missing, use null. "
        "IMPORTANT: 'platform' must be a content creation platform like 'YouTube', 'Instagram', 'TikTok', 'Twitch'. "
        "Do NOT use email providers (gmail) or plan names (Basic/Pro) as the platform. If the user hasn't explicitly stated a platform, return null."
        "IMPORTANT: Output ONLY valid JSON."
    )
    
    # We pass full history for context extraction to catch early names
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[:]])
    
    extraction_response = llm.invoke([
        SystemMessage(content=extraction_prompt), 
        HumanMessage(content=f"History:\n{history_str}")
    ])
    
    print(f"[DEBUG] Extraction Raw: {extraction_response.content}") # Debugging
    
    import json
    try:
        # cleanup markdown json if present
        content = extraction_response.content.replace("```json", "").replace("```", "").strip()
        # Find json brace start/end in case of extra text
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
             content = content[start:end]
             
        data = json.loads(content)
        print(f"[DEBUG] Extracted Data: {data}") # Debugging
        content = extraction_response.content.replace("```json", "").replace("```", "").strip()
        # Find json brace start/end in case of extra text
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
             content = content[start:end]
             
        data = json.loads(content)
        
        # Merge with existing
        if not lead_info['name'] and data.get('name'): lead_info['name'] = data['name']
        if not lead_info['email'] and data.get('email'): lead_info['email'] = data['email']
        if not lead_info['platform'] and data.get('platform'): lead_info['platform'] = data['platform']
        
        print(f"[DEBUG] Current Lead Info: {lead_info}")
        
    except:
        pass # parsing failed, assume nothing new extracted
        
    # Check what's missing
    missing = []
    if not lead_info['name']: missing.append("your name")
    elif not lead_info['email']: missing.append("your email address")
    elif not lead_info['platform']: missing.append("which platform you create content for")
    
    if not missing:
        # All done -> Trigger tool
        result = mock_lead_capture(lead_info['name'], lead_info['email'], lead_info['platform'])
        return {
            "messages": [AIMessage(content=f"Thanks {lead_info['name']}! I've signed you up. {result}")],
            "lead_info": lead_info,
            "intent": "COMPLETE"
        }
    
    # Ask for the first missing item
    response_msg = f"Great! I just need {missing[0]} to get you started."
    return {"messages": [AIMessage(content=response_msg)], "lead_info": lead_info}


# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("detect_intent", detect_intent)
workflow.add_node("greeting", handle_greeting)
workflow.add_node("inquiry", handle_inquiry)
workflow.add_node("qualify", handle_lead_qualification)

workflow.set_entry_point("detect_intent")

def route_intent(state: AgentState):
    intent = state['intent']
    if intent == 'GREETING':
        return "greeting"
    elif intent == 'PRODUCT_INQUIRY':
        return "inquiry"
    elif intent == 'HIGH_INTENT':
        return "qualify"
    elif intent == 'PROVIDING_INFO':
        return "qualify" # Continue qualification loop
    else:
        return "inquiry"

workflow.add_conditional_edges(
    "detect_intent",
    route_intent,
    {
        "greeting": "greeting",
        "inquiry": "inquiry",
        "qualify": "qualify"
    }
)

workflow.add_edge("greeting", END)
workflow.add_edge("inquiry", END)
workflow.add_edge("qualify", END) 

app = workflow.compile()
