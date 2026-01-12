# AutoStream Conversational Agent

A real-world GenAI agent for AutoStream, designed to handle inquiries, detect high-intent leads, and capture contact information using a RAG-powered knowledge base and stateful conversation flow.

## 1. How to Run Locally

### Prerequisites
- Python 3.9+
- A Google Cloud Project with the Gemini API enabled (or compatible API key).

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd AutoStreamAgent
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure Environment:
    - Create a `.env` file in the root directory.
    - Add your API Key:
      ```
      GOOGLE_API_KEY=your_actual_api_key
      ```

### Execution
Run the CLI agent:
```bash
python main.py
```

## 2. Architecture Explanation

### Why LangGraph?
We chose **LangGraph** over AutoGen for this specific use case because:
1.  **Deterministic Control**: The lead qualification flow requires a strict, predictable sequence (Ask Name -> Ask Email -> Ask Platform). LangGraph's state machine approach provides precise control over these transitions compared to AutoGen's more conversational/autonomous multi-agent nature.
2.  **Shared State**: LangGraph's `StateGraph` makes it easy to maintain a single source of truth (`AgentState`) for the collected lead information and conversation history.

### State Management
State is managed using a `TypedDict` Schema (`AgentState`) passed between nodes:
- **Conversation History**: Retained in the `messages` list, utilizing `operator.add` to accumulate turns effectively.
- **Contextual Data**: Specialized fields (`intent`, `lead_info`) track the current conversation phase and extracted entities across turns.
- **Persistence**: In this CLI implementation, state is held in memory during the process lifecycle. For production, LangGraph Checkpointers (e.g., Postgres, Redis) allows resuming sessions across disconnected web requests.

## 3. WhatsApp Deployment Strategy

To deploy this agent on WhatsApp:

1.  **WhatsApp Business API**: Set up a Meta App with the WhatsApp product.
2.  **Webhook Endpoint**: Create a web server (using FastAPI or Flask) to receive incoming messages via Webhooks.
3.  **Integration Logic**:
    - When a webhook event arrives, extract the `user_message` and `sender_phone`.
    - Retrieve the persisted LangGraph thread associated with `sender_phone` (acting as the session ID).
    - Run `app.invoke()` with the new message.
    - Send the `result['messages'][-1].content` back to the user via the WhatsApp API (`POST /messages`).
4.  **Async Handling**: Ensure the webhook acknowledges the request immediately (200 OK) and processes the diverse agent logic in a background task to prevent timeouts.

## 4. File Structure
- `agent.py`: Core logic, graph definition, nodes, and tool implementation.
- `main.py`: CLI entry point and event loop.
- `data/knowledge_base.md`: Static data for RAG.
- `requirements.txt`: Project dependencies.
