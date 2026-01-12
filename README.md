# AutoStream Conversational Agent

A real-world GenAI agent for AutoStream, designed to handle inquiries, detect high-intent leads, and capture contact information using a RAG-powered knowledge base and stateful conversation flow.

## 1. How to Run Locally

### Prerequisites
- Python 3.9+
- **Ollama** installed locally (to run the Llama 3 model).

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
3.  **Setup Local Model**:
    - Install [Ollama](https://ollama.com/).
    - Pull the model:
      ```bash
      ollama pull llama3
      ```
    - Start the Ollama server:
      ```bash
      ollama serve
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



## 4. File Structure
- `agent.py`: Core logic, graph definition, nodes, and tool implementation.
- `main.py`: CLI entry point and event loop.
- `data/knowledge_base.md`: Static data for RAG.
- `requirements.txt`: Project dependencies.
