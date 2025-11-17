import asyncio
import json
import sys

import streamlit as st
import ollama

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

MODEL_NAME = "gpt-oss:120b-cloud"
SERVER_SCRIPT = "mcp_sales_server.py"

TOOLS_PROMPT_TEMPLATE = """
You are an AI assistant that can call tools via the Model Context Protocol (MCP).

You have access to these tools:

{tools_description}

STRICT RULES (READ CAREFULLY):

1. You MUST output ONLY ONE tool call at a time.
2. Your response MUST be exactly ONE JSON object:
   {"tool": "<tool_name>", "args": {...}}
3. NEVER output more than one JSON object.
4. NEVER include any text, explanation, or reasoning outside the JSON.
5. If multiple steps are needed, output ONE tool call, wait for the result,
   then request the next tool in the next message.
6. If no tool is needed, reply with normal natural language (no JSON).
"""

EXPLAIN_PROMPT = """
Explain the result of a tool call to a non-technical business user.

User question:
{user_question}

Tool used: {tool_name}

Tool output:
{tool_output}

Explain clearly.
"""

async def get_tools_description(session: ClientSession) -> str:
    resp = await session.list_tools()
    lines = []
    for tool in resp.tools:
        lines.append(f"- {tool.name}: {tool.description} | input_schema={tool.inputSchema}")
    return "\n".join(lines)

async def call_mcp_tool(session: ClientSession, tool_name: str, args: dict) -> dict:
    result = await session.call_tool(tool_name, args)
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    texts = []
    for c in result.content:
        if isinstance(c, types.TextContent):
            texts.append(c.text)
    if texts:
        return {"text": "\n".join(texts)}
    return {"raw": "No usable content returned."}

def ollama_chat(messages):
    """Synchronous Ollama call"""
    try:
        resp = ollama.chat(model=MODEL_NAME, messages=messages)
        return resp["message"]["content"].strip()
    except Exception as e:
        st.error(f"Ollama error: {e}")
        raise Exception(f"Ollama chat failed: {e}")

async def get_mcp_tools() -> str:
    """Get tool descriptions from MCP server"""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_desc = await get_tools_description(session)
            return tools_desc

async def call_mcp_tool_standalone(tool_name: str, args: dict) -> dict:
    """Call a tool via MCP server"""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_output = await call_mcp_tool(session, tool_name, args)
            return tool_output

def run_agent(user_query: str) -> str:
    """Run the agent with separate MCP sessions"""
    try:
        # Step 1: Get tools description (separate MCP session)
        st.write("⏳ Getting tools from MCP server...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tools_desc = loop.run_until_complete(get_mcp_tools())
        loop.close()
        st.write(f"✓ Got tools")
        
        # Step 2: Ask Ollama what to do (no MCP session active)
        system_prompt = TOOLS_PROMPT_TEMPLATE.replace("{TOOLS_DESC}", tools_desc)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        
        st.write("⏳ Asking Ollama...")
        first_reply = ollama_chat(messages)
        st.write(f"✓ Ollama replied")
        
        # Step 3: Parse response
        tool_name = None
        tool_args = None
        try:
            parsed = json.loads(first_reply)
            if isinstance(parsed, dict) and "tool" in parsed:
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})
                st.write(f"✓ Tool call detected: {tool_name}")
        except json.JSONDecodeError:
            st.write("✓ Plain text response")
            return first_reply
        
        if not tool_name:
            return first_reply
        
        # Step 4: Call the tool (new separate MCP session)
        st.write(f"⏳ Calling tool: {tool_name}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tool_output = loop.run_until_complete(call_mcp_tool_standalone(tool_name, tool_args))
        loop.close()
        st.write("✓ Tool executed")
        
        # Step 5: Get final explanation (no MCP session active)
        explain_prompt = EXPLAIN_PROMPT.format(
            user_question=user_query,
            tool_name=tool_name,
            tool_output=json.dumps(tool_output, indent=2),
        )
        explain_messages = [
            {"role": "system", "content": "You are a helpful analytics assistant."},
            {"role": "user", "content": explain_prompt},
        ]
        
        st.write("⏳ Getting explanation...")
        final_answer = ollama_chat(explain_messages)
        st.write("✓ Done!")
        
        return final_answer
        
    except Exception as e:
        st.error(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return f"Error: {str(e)}"

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="MCP + Ollama", page_icon="🤖")
st.title("🤖 MCP + Ollama Sales Forecast Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            answer = run_agent(user_input)
            st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))