import sys
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent import app

def main():
    print("Initialize AutoStream Agent...")
    print("Type 'quit' or 'exit' to end the session.")
    
    # Initialize conversation state
    # We maintain the list of messages effectively acting as memory
    chat_history = []
    
    # We also need to maintain the lead_info across turns if we are just reinvoking the graph
    # However, LangGraph returns the final state. We should use that.
    current_state = {
        "messages": [],
        "intent": None,
        "lead_info": {"name": None, "email": None, "platform": None}
    }

    print("\nBot: Hi! I'm the AutoStream assistant. How can I help you today?\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower().strip() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break
            
            # Append user message to state
            current_state["messages"].append(HumanMessage(content=user_input))
            
            # Invoke the graph
            # Note: We pass the *entire* state, but LangGraph defaults to *adding* messages if configured with operator.add
            # But here `messages` in input acts as new messages if we initialized it differently, 
            # In our Schema: `messages: Annotated[List[BaseMessage], operator.add]`
            # So passing the whole history in "messages" key might duplicate if we are not careful with how we handle the result.
            # actually, standard pattern with operator.add is to pass ONLY NEW messages.
            # But we also need to pass the *current* values of 'lead_info'.
            
            inputs = {
                "messages": current_state["messages"],
                "lead_info": current_state["lead_info"]
            }
            
            # Run the graph
            result = app.invoke(inputs)
            
            # Update our local state tracking
            # result['messages'] contains the FULL history because of operator.add in the graph state definition?
            # Let's check agent.py again.
            # Yes: `messages: Annotated[List[BaseMessage], operator.add]` means the output will have the full list.
            
            current_state = result
            
            # Get the last message which should be the AI response
            last_msg = current_state["messages"][-1]
            print(f"Bot: {last_msg.content}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
