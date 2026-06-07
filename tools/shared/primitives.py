from typing import Any, Dict, List


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _norm(s: Any) -> str:
    return str(s or "").strip()


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp_nonneg_int(x: Any) -> int:
    v = _safe_int(x, 0)
    return 0 if v < 0 else v


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _subtasks_brief_task_only(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for st in subtasks or []:
        out.append({
            "id": st.get("id"),
            "name": st.get("name"),
            "description": st.get("description"),
            "task": st.get("task"),
            "capability_target": st.get("capability_target"),
            "domain": st.get("domain"),
            "robustness_requirement": st.get("robustness_requirement")
        })
    return out
