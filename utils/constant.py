import os
from dotenv import load_dotenv

load_dotenv()


def str_to_bool(value):
    """convert string to bool"""
    true_values = {'true', 'yes', '1', 'on', 't', 'y'}
    false_values = {'false', 'no', '0', 'off', 'f', 'n'}

    if isinstance(value, bool):
        return value

    if not value:
        return False

    value = str(value).lower().strip()
    if value in true_values:
        return True
    if value in false_values:
        return False
    return True  # default return True


API_BASE_URL = os.getenv('API_BASE_URL', "https://api.bltcy.ai/v1")
API_KEY = os.getenv("LLM_API_KEY", "")

NOT_SUPPORT_SENDER = ["mistral", "groq"]
MUST_ADD_USER = ["deepseek/deepseek-reasoner", "o1-mini"]
NOT_SUPPORT_FN_CALL = ["o1-mini", "deepseek/deepseek-reasoner"]
NOT_USE_FN_CALL = ["deepseek/deepseek-chat"] + NOT_SUPPORT_FN_CALL

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
# Agent cache: context keys not written (re-injected by flow on load)
CONTEXT_KEYS_EXCLUDED_FROM_CACHE = frozenset({
    "id2card",
    "model_config_path",
    "tools_list",
})

GROUNDING_STAGE_KEYS: tuple = (
    "dataset_preference", "retrieval_result", "retrieval_searched",
    "selected_candidate_ids", "candidate_selection_done",
    "transformability", "scored_candidates", "scored_status",
)
