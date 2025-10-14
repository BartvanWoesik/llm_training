import asyncio
import json
import os
import traceback
import uuid

from dotenv import load_dotenv

import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

# Directory to store chat history
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

# MCP config
mcp_config = {
    "schedule": {
        "url": "http://127.0.0.1:8000/mcp",
        "transport": "sse",
       
    },
    "reservations": {
        "url": "http://127.0.0.1:8001/mcp",
        "transport": "sse",
       
    }
}

# Session state for chat history and agent
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)


def get_azure_openai():
    return AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    openai_api_type="azure",
    api_version="2024-12-01-preview",
    azure_endpoint="https://i4talent-openai.openai.azure.com",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

async def setup_agent():
    print("Strarting up agent")
    llm = get_azure_openai()
    mcpc = MultiServerMCPClient(mcp_config)
    tools = await mcpc.get_tools()
    if not tools:
        st.error("No tools discovered from the MCP server.")
        return None
    agent_executor = create_react_agent(llm, tools)
    return agent_executor

async def agent_respond(user_input):
    memory = st.session_state.memory
    history = memory.load_memory_variables({}).get("history", [])
    messages = history + [("user", user_input)]
    agent_executor = st.session_state.agent_executor
    final_answer = ""


    async for event in agent_executor.astream_events({"messages": messages}, version="v1"):
        kind = event["event"]
        if kind == "on_tool_start":
            tool_name = event.get("name", "Unknown Tool")
            st.info(f"Agent is using tool: {tool_name}")
        elif kind == "on_tool_end":
            tool_name = event.get("name", "Unknown Tool")
            st.success(f"Tool usage completed: {tool_name}")

    streaming_placeholder = st.empty()

    async for event in agent_executor.astream_events({"messages": messages}, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                final_answer += content
                streaming_placeholder.text(final_answer)  

    memory.save_context({"input": user_input}, {"output": final_answer})
    return final_answer

def run_async(coro):
    """
    Run async coroutine in Streamlit, handling event loop issues.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(coro)
            asyncio.set_event_loop(loop)  
            return result
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

def format_exception_group(e):
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    details = [tb]
    if hasattr(e, "exceptions"):
        for idx, sub in enumerate(e.exceptions):
            details.append(f"\n--- Sub-exception {idx+1} ---\n")
            details.append("".join(traceback.format_exception(type(sub), sub, sub.__traceback__)))
    return "\n".join(details)

def show_chat():
    for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

def start_messages():
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your MCP agent. How can I assist you today?"
        })

def save_chat_history(session_id):
    """Save the current chat history to a file."""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    with open(file_path, "w") as f:
        json.dump(st.session_state.messages, f)

def load_chat_history(session_id):
    """Load chat history from a file."""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            st.session_state.messages = json.load(f)
    else:
        st.error(f"Chat session '{session_id}' not found.")

def list_chat_sessions():
    """List all saved chat sessions."""
    return [f[:-5] for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]

def setup_chatbot():
    st.title("MCP Chatbot")
    st.write("Chat with your MCP agent below.")
    start_messages()

    if st.session_state.agent_executor is None:
        with st.spinner("Setting up agent..."):
            try:
                agent_executor = run_async(setup_agent())
            except Exception as e:
                st.error(f"Agent setup failed: {e}")
                st.exception(e)
                st.text(format_exception_group(e))
                st.stop()
            st.session_state.agent_executor = agent_executor
            if agent_executor is None:
                st.stop()

def manage_sessions():
    """Sidebar for managing chat sessions."""
    with st.sidebar:
        st.header("Chat Sessions")

        if st.button("Start New Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_name = None
            start_messages()
            save_chat_history(st.session_state.session_id)

        st.write("Saved Chat Sessions:")
        for saved_session in list_chat_sessions():
            if st.button(f"Load {saved_session}"):
                load_chat_history(saved_session)

def get_awnser_from_agent(user_input):
    if "session_id" not in st.session_state:
        st.error("No active session. Please start a new chat.")
        return

    if "session_name" not in st.session_state:
        st.session_state.session_name = user_input[:10].strip()

    st.session_state.messages.append({"role": "user", "content": user_input})
    show_chat()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = run_async(agent_respond(user_input))
            except Exception as e:
                st.error(f"Agent error: {e}")
                return
            # DO NOT SHOW CHAT AGAIN BECAUSE THE AGENT STREAMS THE TEXT
            st.session_state.messages.append({"role": "assistant", "content": answer})
    save_chat_history(st.session_state.session_id)
    st.rerun()

def main():
    manage_sessions()
    setup_chatbot()

    user_input = st.chat_input("Type your message...")
    if not user_input:
        show_chat()
        return

    get_awnser_from_agent(user_input)

if __name__ == "__main__":
    main()
