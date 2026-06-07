from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
from functools import lru_cache


_DEFAULT_IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_DEFAULT_MEDIA_DIR_HINTS: Tuple[str, ...] = (
    "images", "image", "img", "imgs",
    "annotation", "annotations", "ann",
    "mask", "masks", "seg", "segs", "segmentation",
    "label", "labels", "gt", "groundtruth",
    "sounds", "depths", "sketch",
    "target_image", "target_images", "source_images",
)


def _norm_basename(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    s = p.strip().replace("\\", "/")
    if not s:
        return ""
    return os.path.basename(s).strip().lower()


def _dedupe_keep_order(xs: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _iter_parent_dirs(path: str, max_levels: int = 2) -> List[str]:
    out: List[str] = []
    cur = os.path.abspath(path) if path else ""
    for _ in range(max_levels + 1):
        if not cur:
            break
        out.append(cur)
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        cur = parent
    return _dedupe_keep_order(out)


def _list_subdirs_safe(d: str) -> List[str]:
    try:
        names = os.listdir(d)
    except Exception:
        return []
    out: List[str] = []
    for name in names:
        p = os.path.join(d, name)
        try:
            if os.path.isdir(p):
                out.append(p)
        except Exception:
            continue
    return out


def _build_candidate_dirs(base_dir: str) -> List[str]:
    if not base_dir:
        return []
    base_dir = os.path.abspath(base_dir)
    cand: List[str] = [base_dir]

    for name in _DEFAULT_MEDIA_DIR_HINTS:
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            cand.append(p)

    images_dir = None
    for variant in ("images", "Images", "IMAGES"):
        p = os.path.join(base_dir, variant)
        if os.path.isdir(p):
            images_dir = p
            break
    if images_dir:
        parent = os.path.dirname(images_dir)
        for name in _DEFAULT_MEDIA_DIR_HINTS:
            p = os.path.join(parent, name)
            if os.path.isdir(p):
                cand.append(p)
        for p in _list_subdirs_safe(parent):
            bn = os.path.basename(p).lower()
            if any(h in bn for h in _DEFAULT_MEDIA_DIR_HINTS):
                cand.append(p)

    return _dedupe_keep_order(cand)


@lru_cache(maxsize=32)
def _index_media_files(search_root: str, exts: Tuple[str, ...], max_depth: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not search_root or not os.path.isdir(search_root):
        return out

    root = os.path.abspath(search_root)
    exts_l = tuple(e.lower() for e in exts if isinstance(e, str))

    for dirpath, dirnames, filenames in os.walk(root):
        try:
            rel = os.path.relpath(dirpath, root)
        except Exception:
            rel = "."
        depth = 0 if rel in (".", "") else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue

        for fn in filenames:
            if not isinstance(fn, str):
                continue
            low = fn.lower()
            if exts_l and not any(low.endswith(ext) for ext in exts_l):
                continue
            if low not in out:
                out[low] = os.path.join(dirpath, fn)

    return out


@lru_cache(maxsize=1)
def _load_image_root_path_config() -> Dict[str, Any]:
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cfg_path = os.path.join(repo_root, "utils", "resources", "image_root_path.json")
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _get_image_root_candidate_dirs(dataset_id: Optional[str] = None) -> List[str]:
    cfg = _load_image_root_path_config()
    root_path = cfg.get("root_path")
    if not isinstance(root_path, str) or not root_path.strip():
        return []
    root_path = root_path.strip()

    def _expand_prefixes(prefixes: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(prefixes, list):
            return out
        for p in prefixes:
            if not isinstance(p, str) or not p.strip():
                continue
            d = os.path.join(root_path, p.strip())
            if os.path.isdir(d):
                out.append(os.path.abspath(d))
        return out

    if isinstance(dataset_id, str) and dataset_id.strip() and dataset_id in cfg:
        return _dedupe_keep_order(_expand_prefixes(cfg.get(dataset_id)))

    all_dirs: List[str] = []
    for k, v in cfg.items():
        if k == "root_path":
            continue
        all_dirs.extend(_expand_prefixes(v))
    return _dedupe_keep_order(all_dirs)


def resolve_image_paths(
    image_paths: List[str],
    *,
    dataset_json_path: Optional[str] = None,
    dataset_id: Optional[str] = None,
    exts: Tuple[str, ...] = _DEFAULT_IMAGE_EXTS,
    max_walk_depth: int = 1,
    strict: bool = True,
    max_parent_levels: int = 1,
) -> List[str]:
    if not isinstance(image_paths, list) or not image_paths:
        return []

    resolved_by_key: Dict[str, str] = {}
    basenames: List[str] = []

    for raw in image_paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        raw = raw.strip()
        bn = os.path.basename(raw)
        if not bn:
            continue
        basenames.append(bn)
        try:
            if os.path.exists(raw):
                if bn not in resolved_by_key:
                    resolved_by_key[bn] = os.path.abspath(raw)
        except Exception:
            continue

    if not basenames:
        return []

    candidate_dirs: List[str] = []

    if isinstance(dataset_json_path, str) and dataset_json_path.strip():
        json_dir = os.path.dirname(os.path.abspath(dataset_json_path))
        parent_dir = os.path.dirname(json_dir) if json_dir else ""

        def _collect_media_dirs(base: str) -> None:
            if not base or not os.path.isdir(base):
                return
            for name in _DEFAULT_MEDIA_DIR_HINTS:
                sub = os.path.join(base, name)
                if os.path.isdir(sub):
                    candidate_dirs.append(sub)

        _collect_media_dirs(json_dir)
        if parent_dir and parent_dir != json_dir:
            _collect_media_dirs(parent_dir)

    for d in _get_image_root_candidate_dirs(dataset_id=dataset_id):
        candidate_dirs.append(d)

    seen_dirs = set()
    final_candidate_dirs: List[str] = []
    for d in candidate_dirs:
        if d not in seen_dirs:
            seen_dirs.add(d)
            final_candidate_dirs.append(d)

    unresolved = [b for b in basenames if b not in resolved_by_key]

    if final_candidate_dirs and unresolved:
        for b in list(unresolved):
            found_path = None
            for d in final_candidate_dirs:
                p = os.path.join(d, b)
                try:
                    if os.path.exists(p):
                        found_path = os.path.abspath(p)
                        break
                except Exception:
                    continue
            if found_path:
                resolved_by_key[b] = found_path
                unresolved.remove(b)

    result: List[str] = []
    for raw in image_paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        bn = os.path.basename(raw.strip())
        path = resolved_by_key.get(bn)
        if path:
            result.append(path)

    return result
