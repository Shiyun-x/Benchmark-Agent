import json
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message
from typing import Any, Dict, List, Callable, Union, Optional, Tuple

# Third-party imports
from pydantic import BaseModel

AgentFunction = Callable[[], Union[str, "Agent", dict]]


def _messages_to_tool_trace(
    messages: List[Dict[str, Any]],
    max_arg_len: int = 400,
    max_result_len: int = 600,
) -> List[Dict[str, Any]]:
    """Extract tool call + result pairs from agent messages for UI display."""
    trace: List[Dict[str, Any]] = []
    pending_calls: Dict[str, Dict[str, Any]] = {}
    for msg in messages or []:
        role = (msg.get("role") or "").strip().lower()
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tid = (tc.get("id") or "").strip()
                fn = tc.get("function") or {}
                name = (fn.get("name") or "").strip()
                args_raw = (fn.get("arguments") or "").strip()
                if not tid or not name:
                    continue
                try:
                    args_obj = json.loads(args_raw) if args_raw else {}
                    args_preview = json.dumps(args_obj, ensure_ascii=False, indent=2)
                except Exception:
                    args_preview = args_raw
                if len(args_preview) > max_arg_len:
                    args_preview = args_preview[:max_arg_len] + "\n…"
                pending_calls[tid] = {"name": name, "arguments": args_preview}
        elif role == "tool":
            tid = (msg.get("tool_call_id") or "").strip()
            name = (msg.get("name") or "").strip()
            content = (msg.get("content") or "").strip()
            if len(content) > max_result_len:
                content = content[:max_result_len] + "\n…"
            if tid and tid in pending_calls:
                entry = {**pending_calls.pop(tid), "result": content}
                trace.append(entry)
            elif name:
                trace.append({"name": name, "arguments": "", "result": content})
    for _tid, call in pending_calls.items():
        trace.append({**call, "result": "(no result)"})
    return trace


def _messages_to_design_agent_history(
    messages: List[Dict[str, Any]],
    max_reason_len: int = 240,
    max_arg_len: int = 500,
    max_result_len: int = 700,
) -> List[Dict[str, Any]]:
    """Extract concise action history from one completed agent run."""
    history: List[Dict[str, Any]] = []
    pending_calls: Dict[str, Dict[str, Any]] = {}
    for msg in messages or []:
        role = (msg.get("role") or "").strip().lower()
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tid = (tc.get("id") or "").strip()
                fn = tc.get("function") or {}
                name = (fn.get("name") or "").strip()
                args_raw = (fn.get("arguments") or "").strip()
                if not tid or not name:
                    continue
                try:
                    args_obj = json.loads(args_raw) if args_raw else {}
                except Exception:
                    args_obj = {}
                decision_rationale = str(args_obj.pop("decision_rationale", "") or "").strip()
                if len(decision_rationale) > max_reason_len:
                    decision_rationale = decision_rationale[:max_reason_len] + "..."
                try:
                    args_preview = json.dumps(args_obj, ensure_ascii=False, indent=2)
                except Exception:
                    args_preview = str(args_obj)
                if len(args_preview) > max_arg_len:
                    args_preview = args_preview[:max_arg_len] + "\n..."
                pending_calls[tid] = {
                    "tool": name,
                    "decision_rationale": decision_rationale,
                    "args": args_preview,
                }
        elif role == "tool":
            tid = (msg.get("tool_call_id") or "").strip()
            name = (msg.get("name") or "").strip()
            result = str(msg.get("content") or "").strip()
            if len(result) > max_result_len:
                result = result[:max_result_len] + "\n..."
            if tid and tid in pending_calls:
                entry = {**pending_calls.pop(tid), "result": result}
                history.append(entry)
            elif name:
                history.append({"tool": name, "decision_rationale": "", "args": "{}", "result": result})
    for _tid, call in pending_calls.items():
        history.append({**call, "result": "(no result)"})
    return history


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-5-mini"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = False
    examples: Union[List[Tuple[dict, str]], Callable[[], str]] = []
    handle_mm_func: Callable[[], str] = None


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
    image: Optional[str] = None # base64 encoded image
