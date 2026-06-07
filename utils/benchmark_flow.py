"""
Flow and caching: ToolModule / AgentModule wrap single-step calls with caching.
- Tool: caches (args, result); on cache hit, returns result directly.
- Agent: caches (messages, context_variables); if context_output_keys is configured, only persists keys updated by that agent.
"""
import json
import os
import yaml

from utils.util import single_select_menu
from utils.core import MetaChain
from utils.logger import MetaChainLogger
from utils.agent_utils import Agent, _messages_to_design_agent_history
from utils.constant import GROUNDING_STAGE_KEYS

from typing import Union, Dict, List, Callable, Any, Optional, Iterable
from abc import ABC, abstractmethod

_FORCE_NO_CACHE_FOR_RUN = False


def _reset_cache_policy() -> None:
    global _FORCE_NO_CACHE_FOR_RUN
    _FORCE_NO_CACHE_FOR_RUN = False


def _enable_force_no_cache() -> None:
    global _FORCE_NO_CACHE_FOR_RUN
    _FORCE_NO_CACHE_FOR_RUN = True


def _is_force_no_cache() -> bool:
    return _FORCE_NO_CACHE_FOR_RUN


def _normalize_agent_name(agent_name: str) -> str:
    return (agent_name or "").replace(" ", "_").lower()


def _load_agent_context_output_keys(config_path: str, agent_name: str) -> Union[tuple, None]:
    """Load context_output_keys for the named agent from agent_cache_config.yaml."""
    key = _normalize_agent_name(agent_name)
    if not key:
        return None
    path = config_path
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    mapping = data.get("context_output_keys")
    if not isinstance(mapping, dict):
        return None
    val = mapping.get(key)
    if val is None:
        return None
    if isinstance(val, list) and len(val) == 0:
        return None
    if isinstance(val, list):
        return tuple(str(k) for k in val)
    return None


def _clear_grounding_runtime_state(context_variables: Dict) -> None:
    """
    Clear per-subtask grounding progress so Grounding Agent can re-run end-to-end.
    Keep design outputs (id/name/description/etc.) untouched.
    """
    subtasks = list(context_variables.get("subtasks", []) or [])
    removable_keys = {
        "dataset_preference",
        "retrieval_result",
        "retrieval_searched",
        "selected_candidate_ids",
        "candidate_selection_done",
        "transformability",
        "scored_candidates",
        "scored_status",
    }
    for st in subtasks:
        if isinstance(st, dict):
            for k in removable_keys:
                st.pop(k, None)
    context_variables["subtasks"] = subtasks
    context_variables.pop("grounding_result", None)
    context_variables.pop("grounding_feedback", None)
    # Keep grounding stage cache enabled by default.
    # Stage tools decide per-file whether to load existing cache or recompute.
    context_variables["_use_grounding_stage_cache"] = True


def _clear_agent_runtime_state_for_fresh_run(agent_name: str, context_variables: Dict) -> None:
    agent_norm = _normalize_agent_name(agent_name)
    if agent_norm == "design_agent":
        # Preserve upstream task description fields, but reset design products.
        context_variables.pop("design_result", None)
        context_variables.pop("design_tool_trace", None)
        context_variables["subtasks"] = []
        context_variables["proposed_subtasks"] = []
    elif agent_norm == "grounding_agent":
        _clear_grounding_runtime_state(context_variables)
    elif agent_norm == "allocation_agent":
        context_variables.pop("last_allocation", None)
        context_variables.pop("allocation", None)
        context_variables.pop("allocation_result", None)
        subtasks = list(context_variables.get("subtasks", []) or [])
        for st in subtasks:
            if isinstance(st, dict):
                for k in ("quota_left", "assigned_dataset_ids"):
                    st.pop(k, None)
        context_variables["subtasks"] = subtasks



def _sanitize_subtasks_for_cache(ctx: Dict, stage: str) -> Dict:
    """Strip subtask fields irrelevant to the given pipeline stage before caching."""
    subtasks = list(ctx.get("subtasks", []) or [])
    if not subtasks:
        return ctx
    base_keys = {
        "id", "name", "description", "answer_type",
        "modalities", "sample_schema", "keywords", "notes",
    }
    grounding_keys = set(GROUNDING_STAGE_KEYS)
    allocation_keys = {"importance", "quota", "quota_left", "assigned_dataset_ids"}
    if stage == "design":
        allowed = base_keys
    elif stage == "grounding":
        allowed = base_keys | grounding_keys
    elif stage == "allocation":
        allowed = base_keys | grounding_keys | allocation_keys
    else:
        allowed = base_keys | grounding_keys | allocation_keys
    ctx["subtasks"] = [
        {k: v for k, v in st.items() if k in allowed}
        for st in subtasks if isinstance(st, dict)
    ]
    return ctx


def _make_postprocess_fn(agent_name: str, stage: str) -> Callable:
    """Return a postprocess closure that appends run history and sanitizes subtask cache fields."""
    history_key = f"{agent_name}_agent_history"

    def _postprocess(messages: List[Dict], ctx: Dict) -> Dict:
        run_history = _messages_to_design_agent_history(messages)
        existing = ctx.get(history_key) or []
        if run_history:
            if existing[-len(run_history):] != run_history:
                existing = existing + run_history
            ctx[history_key] = existing[-40:]
        ctx = _sanitize_subtasks_for_cache(ctx, stage=stage)
        return ctx

    return _postprocess


class AgentModule:
    def __init__(
        self,
        agent: Agent,
        client: MetaChain,
        cache_path: str,
        exclude_context_keys_from_cache: set = None,
        context_output_keys: Optional[Iterable[str]] = None,
        postprocess_fn: Optional[Callable[[List[Dict], Dict], Dict]] = None,
    ):
        self.agent = agent
        self.client = client
        self.cache_path = cache_path
        self.exclude_context_keys_from_cache = exclude_context_keys_from_cache or set()
        self.context_output_keys = set(context_output_keys) if context_output_keys is not None else None
        self.postprocess_fn = postprocess_fn

    def _postprocess(self, ret_messages: List[Dict], ret_ctx: Dict) -> Dict:
        if self.postprocess_fn is not None:
            return self.postprocess_fn(ret_messages, ret_ctx)
        return ret_ctx
    async def __call__(
        self,
        messages: List[Dict],
        context_variables: Dict,
        iter_times: int = None,
        cache_name: str = None,
        force_recompute: bool = False,
        *args,
        **kwargs,
    ):
        # messages = [{"role": "user", "content": query}]
        if force_recompute:
            _enable_force_no_cache()
            response = await self.client.run_async(self.agent, messages, context_variables=context_variables, debug=True)
            ret_messages = response.messages
            ret_context_variables = response.context_variables
            if ret_messages[-1]["role"] != "error":
                ret_messages.append({"role": "success", "content": "The agent successfully generated a response."})
            ret_context_variables = self._postprocess(ret_messages[:-1], ret_context_variables)
            # Always overwrite the same cache target.
            self.save_cache(self.agent.name, ret_messages[:-1], iter_times, ret_context_variables, cache_name=cache_name)
            messages.extend(ret_messages[:-1])
            context_variables.update(ret_context_variables)
            if ret_messages[-1]["role"] == "error":
                raise Exception(ret_messages[-1]["content"])
            return messages, context_variables

        cache_file_exists = self._cache_file_exists(self.agent.name, iter_times=iter_times, cache_name=cache_name)
        agent_cache, escape_running = self.check_cache(self.agent.name, iter_times, cache_name=cache_name)
        if agent_cache and escape_running:
            # if use cache
            messages.extend(agent_cache["messages"])
            context_variables.update(agent_cache["context_variables"])
        elif agent_cache and not escape_running:
            # if cache exists but not use
            messages.extend(agent_cache["messages"])
            context_variables.update(agent_cache["context_variables"])
            response = await self.client.run_async(self.agent, messages, context_variables=context_variables, debug=True)
            ret_messages = response.messages
            ret_context_variables = response.context_variables
            if ret_messages[-1]["role"] != "error":
                ret_messages.append({"role": "success", "content": "The agent successfully generated a response."})
            ret_context_variables = self._postprocess(ret_messages[:-1], ret_context_variables)
            self.save_cache(self.agent.name, agent_cache["messages"] + ret_messages[:-1], iter_times, ret_context_variables, cache_name=cache_name)
            messages.extend(ret_messages[:-1])
            context_variables.update(ret_context_variables)
            if ret_messages[-1]["role"] == "error":
                raise Exception(ret_messages[-1]["content"])
        else:
            # cache file exists but user chose "No": recompute from a fresh runtime state
            if cache_file_exists:
                _clear_agent_runtime_state_for_fresh_run(self.agent.name, context_variables)
            # if the cache does not exist
            response = await self.client.run_async(self.agent, messages, context_variables=context_variables, debug=True)
            ret_messages = response.messages
            ret_context_variables = response.context_variables
            if ret_messages[-1]["role"] != "error":
                ret_messages.append({"role": "success", "content": "The agent successfully generated a response."})
            ret_context_variables = self._postprocess(ret_messages[:-1], ret_context_variables)
            self.save_cache(self.agent.name, ret_messages[:-1], iter_times, ret_context_variables, cache_name=cache_name)
            messages.extend(ret_messages[:-1])
            context_variables.update(ret_context_variables)
            if ret_messages[-1]["role"] == "error":
                raise Exception(ret_messages[-1]["content"])
        return messages, context_variables
    def _cache_file_exists(self, agent_name: str, iter_times: int = None, cache_name: str = None) -> bool:
        agent_name_norm = _normalize_agent_name(agent_name)
        if iter_times is not None:
            agent_name_norm = agent_name_norm + f"_iter_{iter_times}"
        if cache_name is not None:
            agent_name_norm = agent_name_norm + "/" + cache_name
        cache_file = f"{self.cache_path}/agents/{agent_name_norm}.json"
        return os.path.exists(cache_file)
    def save_cache(self, agent_name, messages, iter_times: int = None, context_variables: Dict = None, cache_name: str = None):
        agent_name = agent_name.replace(" ", "_").lower()
        if iter_times is not None:
            agent_name = agent_name + f"_iter_{iter_times}"
        if cache_name is not None:
            agent_name = agent_name + "/" + cache_name
        agent_cache_file = f"{self.cache_path}/agents/{agent_name}.json"
        os.makedirs(os.path.dirname(agent_cache_file), exist_ok=True)
        # prefer saving only keys updated by this agent; otherwise save all minus excluded keys
        if context_variables is None:
            ctx_to_save = {}
        elif self.context_output_keys is not None:
            ctx_to_save = {k: context_variables[k] for k in self.context_output_keys if k in context_variables}
        elif self.exclude_context_keys_from_cache:
            ctx_to_save = {k: v for k, v in context_variables.items() if k not in self.exclude_context_keys_from_cache}
        else:
            ctx_to_save = context_variables
        with open(agent_cache_file, "w", encoding="utf-8") as f:
            json.dump({"messages": messages, "context_variables": ctx_to_save}, f, ensure_ascii=False, indent=4)
    def check_cache(self, agent_name, iter_times: int = None, cache_name: str = None):
        """
        check cache
        if cache exists, return cache
        else return None
        """
        agent_name_norm = agent_name.replace(" ", "_").lower()
        if iter_times is not None:
            agent_name_norm = agent_name_norm + f"_iter_{iter_times}"
        if cache_name is not None:
            agent_name_norm = agent_name_norm + "/" + cache_name
        cache_file = f"{self.cache_path}/agents/{agent_name_norm}.json"
        if os.path.exists(cache_file):
            if _is_force_no_cache():
                return None, False
            choice = single_select_menu(["Yes", "Resume", "No"], f"The agent '{agent_name}' cache file exists, do you want to use it?")
            if choice == "Yes":
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f), True
            elif choice == "Resume":
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f), False
            else:
                _enable_force_no_cache()
                return None, False
        return None, False
    
class ToolModule:
    def __init__(self, tool: Callable[[Any], Union[str, Dict]], cache_path: str):
        self.tool = tool
        self.cache_path = cache_path
    def __call__(
        self,
        tool_args: Dict,
        cache_name: str = None,
        force_recompute: bool = False,
        *args,
        **kwargs,
    ):
        if force_recompute:
            tool_result = self.tool(**tool_args)
            self.save_cache(self.tool, tool_args, tool_result, cache_name)
            return tool_result

        tool_cache = self.check_cache(self.tool.__name__, cache_name)
        if tool_cache:
            return tool_cache
        tool_result = self.tool(**tool_args)
        self.save_cache(self.tool, tool_args, tool_result, cache_name)
        return tool_result
    def save_cache(self, tool: Callable, tool_args: Dict, tool_result: Union[str, Dict], cache_name: str = None):
        tool_name = tool.__name__
        if cache_name is not None:
            tool_name = tool_name + "/" + cache_name
        tool_cache_file = f"{self.cache_path}/tools/{tool_name}.json"
        os.makedirs(os.path.dirname(tool_cache_file), exist_ok=True)
        # cache stores result only (not args) to keep files small; use name+args+result for debugging
        tool_cache_dict = {
            "name": tool.__name__,
            "result": tool_result
        }
        with open(tool_cache_file, "w", encoding="utf-8") as f:
            json.dump(tool_cache_dict, f, ensure_ascii=False, indent=4)
    def check_cache(self, tool_name: str, cache_name: str = None):
        if cache_name is not None:
            tool_name = tool_name + "/" + cache_name
        cache_file = f"{self.cache_path}/tools/{tool_name}.json"
        if os.path.exists(cache_file):
            if _is_force_no_cache():
                return None
            choice = single_select_menu(["Yes", "No"], f"The tool '{tool_name}' cache file exists, do you want to use it?")
            if choice == "Yes":
                with open(cache_file, "r", encoding="utf-8") as f:
                    tool_cache_dict = json.load(f)
                    return tool_cache_dict["result"]
            else:
                _enable_force_no_cache()
                return None
        return None

class FlowModule(ABC):
    def __init__(self, cache_path: str, log_path: Union[str, None, MetaChainLogger] = None, model: str = "gpt-5-mini", model_config_path: str = None):
        _reset_cache_policy()
        self.cache_path = cache_path
        self.client = MetaChain(log_path=log_path, model_config_path=model_config_path)
        self.model = model
    @abstractmethod
    async def forward(self, *args, **kwargs):
        raise NotImplementedError("subclass should implement this method")
    
    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    