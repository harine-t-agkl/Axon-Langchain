import traceback
import json
import re
import time
import concurrent.futures
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.registry import TOOLS

# Local LLM
llm = OllamaLLM(model="gemma3:latest", base_url="http://localhost")

# -------------------------------
# REACT-LIKE PROMPT
# -------------------------------

SYSTEM = """
You are an intelligent assistant that can use tools when needed.
pip install bibtexparser langchain-community

You have access to the following tools:
{tool_list}

RULES:
- Think step-by-step.
- If a tool is needed, output ONLY a JSON dict:
  {{"tool": "<toolname>", "input": "<text>"}}
- Otherwise respond normally with your final answer.
"""

HUMAN = "{input}"

# Convert tool registry to readable text
def format_tool_list(tools):
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", HUMAN),
])

parser = StrOutputParser()

chain = prompt | llm | parser


# -------------------------------
# NEW JSON EXTRACTION FIX
# -------------------------------
def extract_json(s: str):
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None


# ------------------------------------------------------------
# AGENT RUNNER
# ------------------------------------------------------------
# Configuration: tune these numbers as needed
TOOL_CALL_TIMEOUT = 100         # seconds per tool call (prevent slow/blocking tools)
AGENT_TOTAL_TIMEOUT = 1000     # seconds total budget for the whole agent run
MAX_SAME_TOOL_CALLS = 3       # abort if the same tool is requested > this many times

def run_agent(question: str) -> str:
    """
    Runs lightweight ReAct-style loop:
    1. Ask LLM what to do
    2. If tool call → run tool (with a per-tool timeout)
    3. Feed result back until final answer
    """

    loop_limit = 5
    last_output = ""
    start_time = time.time()

    # Track how many times each tool was requested in this run
    tool_call_counts = {}

    for _ in range(loop_limit):

        # Check overall time budget
        elapsed = time.time() - start_time
        if elapsed > AGENT_TOTAL_TIMEOUT:
            return f"Agent aborted: exceeded overall timeout of {AGENT_TOTAL_TIMEOUT} seconds."

        # Ask LLM for next step
        ai_msg = chain.invoke({
            "tool_list": format_tool_list(TOOLS),
            "input": question if not last_output else f"{question}\n\nTool result:\n{last_output}"
        })

        # Try to detect JSON tool call using improved extractor
        tool_call = extract_json(ai_msg)

        # ---- If LLM requests a tool ----
        if isinstance(tool_call, dict) and "tool" in tool_call:
            tool_name = tool_call["tool"]
            tool_input = tool_call.get("input", "")
            print(f" LLM requested tool: {tool_name} with input: {tool_input}")

            # Find tool
            tool = next((t for t in TOOLS if t.name == tool_name), None)
            if not tool:
                return f"ERROR: Unknown tool '{tool_name}'"

            # Increment usage count for this tool
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            if tool_call_counts[tool_name] > MAX_SAME_TOOL_CALLS:
                return (f"Agent aborted: tool '{tool_name}' requested more than "
                        f"{MAX_SAME_TOOL_CALLS} times. Possible loop or malformed tool usage.")

            try:
                print(f" Executing tool: {tool_name}")

                # Run the tool with a per-tool timeout so it cannot block forever.
                # We use a short ThreadPoolExecutor for each tool invocation.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(tool.func, tool_input)
                    try:
                        result = future.result(timeout=TOOL_CALL_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        # Tool timed out: inform LLM and let it try another tool
                        last_output = (f"[TOOL TIMEOUT]\nTool '{tool_name}' exceeded "
                                       f"{TOOL_CALL_TIMEOUT}s and was aborted.")
                        print(f" Tool '{tool_name}' timed out after {TOOL_CALL_TIMEOUT}s.")
                        continue

                # Normal successful tool result
                last_output = f"[TOOL RESULT]\n{result}"
                continue

            except Exception as e:
                # Tool raised an exception: show stacktrace to LLM in last_output but continue
                last_output = f"Tool error: {e}\n{traceback.format_exc()}"
                print(f" Tool '{tool_name}' raised an exception: {e}")
                continue

        # ---- Not a tool call → final answer ----
        return ai_msg

    return "Agent exceeded reasoning loop limit."
