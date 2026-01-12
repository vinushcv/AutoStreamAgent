from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uuid
from agent import app as agent_app

app = FastAPI()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session storage
# {session_id: {"messages": [], "lead_info": {...}}}
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    # Helper to redirect to the static HTML
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_msg = request.message
    
    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "intent": None,
            "lead_info": {"name": None, "email": None, "platform": None}
        }
    
    current_state = sessions[session_id]
    
    # Append user message
    current_state["messages"].append(HumanMessage(content=user_msg))
    
    # Prepare inputs for LangGraph
    inputs = {
        "messages": current_state["messages"],
        "lead_info": current_state["lead_info"]
    }
    
    try:
        # Run agent
        result = agent_app.invoke(inputs)
        
        # Update session state with result
        sessions[session_id] = result
        
        # Extract bot response
        bot_response = result["messages"][-1].content
        return ChatResponse(response=bot_response)
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to console for debugging
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
