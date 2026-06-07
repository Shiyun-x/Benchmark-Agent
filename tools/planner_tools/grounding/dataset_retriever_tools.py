from concurrent.futures import ThreadPoolExecutor

from typing import Any, Dict, List, Tuple, Set

from utils.schema.dataset_card import DatasetCard
from collections import Counter
import math
import re
from utils.schema.dataset_card import create_dataset_card_from_raw
from utils.registry import register_tool
# --------------------------
# BM25 (Okapi) implementation
# --------------------------
class BM25Okapi:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_lens = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_lens)/self.N if self.N > 0 else 0.0

        # DF / IDF
        df = Counter()
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] += 1
        self.idf = {}
        for t, f in df.items():
            # Okapi BM25 IDF with +0.5 smoothing
            self.idf[t] = math.log((self.N - f + 0.5) / (f + 0.5) + 1.0)

        # TF
        self.tf = []
        for doc in corpus_tokens:
            self.tf.append(Counter(doc))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0]*self.N
        if not query_tokens or self.N == 0: 
            return scores
        qtf = Counter(query_tokens)
        for i in range(self.N):
            dl = self.doc_lens[i] or 1
            denom = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            s = 0.0
            tf_i = self.tf[i]
            for t, qf in qtf.items():
                if t not in self.idf:
                    continue
                idf = self.idf[t]
                f = tf_i.get(t, 0)
                if f == 0:
                    continue
                # standard BM25 term
                s += idf * (f * (self.k1 + 1)) / (f + denom)
            scores[i] = s
        return scores

# --------------------------
# TF-IDF cosine (sparse)
# --------------------------
class TfidfIndex:
    def __init__(self, corpus_tokens: List[List[str]]):
        self.N = len(corpus_tokens)
        self.doc_tf = []
        df = Counter()
        for doc in corpus_tokens:
            tf = Counter(doc)
            self.doc_tf.append(tf)
            for t in tf.keys():
                df[t] += 1

        # compute idf
        self.idf = {}
        for t, f in df.items():
            self.idf[t] = math.log((self.N + 1) / (f + 1)) + 1.0  # smooth idf

        # precompute doc norms
        self.doc_norm = []
        for tf in self.doc_tf:
            s = 0.0
            for t, f in tf.items():
                s += (f * self.idf.get(t, 0.0))**2
            self.doc_norm.append(math.sqrt(s) if s > 0 else 1.0)

    def query_vec(self, query_tokens: List[str]) -> Dict[str, float]:
        qtf = Counter(query_tokens)
        qv = {}
        for t, f in qtf.items():
            qv[t] = f * self.idf.get(t, 0.0)
        # l2 normalize
        norm = math.sqrt(sum(v*v for v in qv.values())) or 1.0
        for t in list(qv.keys()):
            qv[t] /= norm
        return qv

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if self.N == 0:
            return []
        qv = self.query_vec(query_tokens)
        scores = [0.0]*self.N
        if not qv:
            return scores
        for i in range(self.N):
            # cosine = dot(q, d) / (||q|| * ||d||); q already normalized
            # compute dot(q, d)
            dot = 0.0
            tf = self.doc_tf[i]
            for t, qw in qv.items():
                if t in tf:
                    dot += qw * (tf[t] * self.idf.get(t, 0.0))
            scores[i] = dot / (self.doc_norm[i] or 1.0)
        return scores


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Simple English/alphanumeric tokenization is sufficient
    return re.findall(r"[a-zA-Z0-9]+", text.lower())




def _normalize_modalities(mods: Any) -> Set[str]:
    """
    Unify various modalities writing styles into a set of lowercase strings.
    Support str / list / tuple / set; other types are ignored.
    """
    out: Set[str] = set()
    if isinstance(mods, str):
        out.add(mods.lower())
    elif isinstance(mods, (list, tuple, set)):
        for m in mods:
            if isinstance(m, str):
                out.add(m.lower())
    return out

HEAVY_MODALITIES: Set[str] = {"audio", "image"}
def is_modality_compatible(subtask: Dict[str, Any], card: "DatasetCard") -> bool:

    sub_mods = _normalize_modalities(subtask.get("modalities") or [])
    io_schemas = card.io_schemas[0]
    
    in_mods = _normalize_modalities(io_schemas.get("in") or io_schemas.get("in_"))
    out_mods = _normalize_modalities(io_schemas.get("out") or io_schemas.get("out_"))
    
    modalities = _normalize_modalities(getattr(card, "modalities", []) or [])
    ds_mods = modalities | in_mods | out_mods
    # no modality information, no hard filter
    norm = {"images":"image","image":"image","audio":"audio","text":"text"}
    ds_mods = set([norm.get(x, x) for x in ds_mods])
    sub_mods = set([norm.get(x, x) for x in sub_mods])

    if not sub_mods or not ds_mods:
        return True

    sub_heavy = sub_mods & HEAVY_MODALITIES
    ds_heavy = ds_mods & HEAVY_MODALITIES

    # ---- Case 0: subtask does not involve any heavy -> dataset cannot have heavy ----
    if not sub_heavy:
        return not bool(ds_heavy)

    # ---- Case A: only involve audio ----
    if sub_heavy == {"audio"}:
        if ds_heavy - {"audio"}:
            return False

        # at least one of audio or text: only video etc. is not matched
        if ("audio" in ds_mods) or ("text" in ds_mods):
            return True
        return False

    # ---- Case B: only involve image ----
    if sub_heavy == {"image"}:
        if "image" not in ds_mods:
            return False
        if "audio" in ds_heavy:
            return False
        return True

    # ---- Case D: subtask involves multiple heavy (audio+images etc.)
    # Subtask involves both image and audio; image is required
    if sub_heavy == {"image", "audio"}:
        if "image" not in ds_mods:
            return False
        return True
    return True


def compute_meta_richness(fields: Dict[str, Any]) -> float:
    """
    Estimate meta information richness from discrete content_extent values
    (short/medium/long/mixed); returns a score in [0, 1].
    """

    if not fields:
        return 0.0

    CONTENT_EXTENT_SCORES = {
        "short": 0.2,
        "medium": 0.5,
        "long": 0.9,
        "mixed": 0.7,
    }

    scores = []
    for name,info in fields.items():
        if name in ["question", "answer","options"]:
            continue
        if not isinstance(info, dict):
            continue
        
        ce = info.get("content_extent")
        if isinstance(ce, str):
            ce = ce.lower().strip()
            score = CONTENT_EXTENT_SCORES.get(ce)
            if score is not None:
                scores.append(score)

    if not scores:
        return 0.0

    avg = sum(scores) / len(scores)
    return avg

def build_native_and_meta_corpus(dataset_cards) -> Tuple[
    List[str], List[List[str]], List[List[str]], List[str], List[str]
]:
    """
    Construct two document views for each dataset_card:
    - native_doc: text describing native QA / task / domain
    - meta_doc:   text describing meta.fields + meta_structure

    Return:
      doc_ids: List[dataset_id]
      native_tokens: List[List[str]]
      meta_tokens:   List[List[str]]
      native_docs_raw: List[str]
      meta_docs_raw:   List[str]
    """
    doc_ids: List[str] = []
    native_docs_raw: List[str] = []
    meta_docs_raw: List[str] = []

    for card in dataset_cards:
         
        doc_ids.append(card.dataset_id)

        # --- native view ---
        native_doc = " ".join([
            card.name or "",
            getattr(card, "description", "") or "",
            getattr(card, "card_text", "") or "",
            " ".join(getattr(card, "tasks", []) or []),
            getattr(card, "domain", "") or "",
            " ".join(getattr(card, "modalities", []) or []),
        ])
        native_docs_raw.append(native_doc)

        # --- meta view ---
        meta = getattr(card, "meta", {}) or {}
        fields = meta.get("fields", {}) or {}
        meta_desc = meta.get("meta_structure_description", "") or ""

        pieces = [meta_desc]
        for role, info in fields.items():
            if not isinstance(info, dict):
                continue
            # role name + style/content_extent/semantics
            pieces.append(role)
            style = info.get("style") or ""
            content_extent = info.get("content_extent") or ""
            sem = info.get("semantics") or ""
            pieces.extend([style, content_extent, sem])

        meta_docs_raw.append(" ".join(pieces))

    native_tokens = [tokenize(t) for t in native_docs_raw]
    meta_tokens = [tokenize(t) for t in meta_docs_raw]

    return doc_ids, native_tokens, meta_tokens, native_docs_raw, meta_docs_raw

def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-6:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def soft_native_match(fields: Dict[str, Any]) -> float:
    """
    Score how QA-friendly the dataset's native structure is.
    Type-A datasets (already have question+answer) get a boost.
    """
    roles = set(fields.keys())
    has_q = "question" in roles
    has_a = "answer" in roles
    has_ctx = "context" in roles

    if has_q and has_a:
        return 1.0
    if has_ctx and has_a:
        return 0.7
    if has_ctx:
        return 0.4
    return 0.1


def soft_meta_match(fields: Dict[str, Any], pref: Dict[str, Any]) -> float:
    """
    Soft-score meta fields using LLM preferred_roles.
    Matching roles add points; missing roles are not penalized.
    """
    score = 0.0
    roles = set(fields.keys())
    for role, cfg in (pref.get("preferred_roles") or {}).items():
        if role in roles:
            score += float(cfg.get("weight", 0.5))
    return score

def build_meta_query(pref: Dict[str, Any]) -> str:
    """
    Flatten LLM dataset_preference into a query string for BM25 / TF-IDF
    retrieval against meta_doc.
    """
    pieces: List[str] = []

    # preferred_roles shape:
    # "preferred_roles": {
    #   "context": {
    #       "reason": "...",
    #       "style_preferences": [...],
    #       "content_extent_preferences": [...],
    #       "semantics_hint": "...",
    #       "weight": 1.0
    #   },
    #   ...
    # }
    for role, cfg in (pref.get("preferred_roles") or {}).items():
        pieces.append(role)
        if isinstance(cfg, dict):
            for k in ("style_preferences", "content_extent_preferences"):
                v = cfg.get(k)
                if isinstance(v, list):
                    pieces.extend(v)
            sem_hint = cfg.get("semantics_hint")
            if isinstance(sem_hint, str):
                pieces.append(sem_hint)

    pieces.extend(pref.get("preferred_domains") or [])
    # pieces.extend(pref.get("preferred_modalities") or [])

    # # notes / recall_strategy can be added sparingly
    # notes = pref.get("notes") or ""
    # if isinstance(notes, str):
    #     pieces.append(notes)

    return " ".join(pieces)


@register_tool("retrieve_candidates")
def retrieve_candidates(
    subtask: Dict[str, Any],
    dataset_cards_json: List[Any],          # DatasetCard
    pref: Dict[str, Any],
    top_k: int = 8,
    method: str = "hybrid",            # "bm25" | "tfidf" | "hybrid"
    alpha: float = 0.6,                # BM25 weight when method is hybrid
) -> List[Dict[str, Any]]:
    """
    High-recall initial screening:
    - native: subtask description ↔ dataset native description
    - meta:   LLM preference ↔ dataset meta fields

    Returns:
    [
      {
        "dataset_id": ...,
        "final_score": float,
        "native_score": float,
        "meta_score": float,
        "debug": {...}
      },
      ...
    ]
    """

    if not dataset_cards_json:
        return []
    dataset_cards: List[DatasetCard] = []
    for card_json in dataset_cards_json:
        card = create_dataset_card_from_raw(card_json)
        dataset_cards.append(card)
    # 0) construct double-view corpus
    doc_ids, native_tokens, meta_tokens, native_docs_raw, meta_docs_raw = \
        build_native_and_meta_corpus(dataset_cards)

    # 1) build BM25 / TF-IDF index
    bm25_native = bm25_meta = None
    tfidf_native = tfidf_meta = None

    if method in ("bm25", "hybrid"):
        bm25_native = BM25Okapi(native_tokens)
        bm25_meta = BM25Okapi(meta_tokens)

    if method in ("tfidf", "hybrid"):
        tfidf_native = TfidfIndex(native_tokens)
        tfidf_meta = TfidfIndex(meta_tokens)

    # 2) prepare query
    modalities = subtask.get("modalities") or []
    if isinstance(modalities, (list, tuple, set)):
        modalities_str = " ".join(modalities)
    else:
        modalities_str = str(modalities)

    native_q_text = " ".join([
        subtask.get("name", "") or "",
        subtask.get("description", "") or "",
        subtask.get("domain", "") or "",
        modalities_str,
        subtask.get("task", "") or "",
        " ".join(subtask.get("keywords", []) or []),
    ])
    native_q_tokens = tokenize(native_q_text)


    meta_q_text = build_meta_query(pref)
    meta_q_tokens = tokenize(meta_q_text)

    # 3) calculate native_sem & meta_sem (each view has its own hybrid combination)
    N = len(dataset_cards)
    native_sem_scores = [0.0] * N
    meta_sem_scores = [0.0] * N

    # -- native view --
    bm25_native_scores = [0.0] * N
    tfidf_native_scores = [0.0] * N
    if bm25_native is not None:
        bm25_native_scores = bm25_native.get_scores(native_q_tokens)
    if tfidf_native is not None:
        tfidf_native_scores = tfidf_native.get_scores(native_q_tokens)

    # -- meta view
    bm25_meta_scores = [0.0] * N
    tfidf_meta_scores = [0.0] * N
    if bm25_meta is not None:
        bm25_meta_scores = bm25_meta.get_scores(meta_q_tokens)
    if tfidf_meta is not None:
        tfidf_meta_scores = tfidf_meta.get_scores(meta_q_tokens)

    bm25_native_scores = normalize_scores(bm25_native_scores) if bm25_native is not None else bm25_native_scores
    tfidf_native_scores = normalize_scores(tfidf_native_scores) if tfidf_native is not None else tfidf_native_scores
    bm25_meta_scores = normalize_scores(bm25_meta_scores) if bm25_meta is not None else bm25_meta_scores
    tfidf_meta_scores = normalize_scores(tfidf_meta_scores) if tfidf_meta is not None else tfidf_meta_scores

    for i in range(N):
        if method == "bm25":
            native_sem_scores[i] = bm25_native_scores[i]
            meta_sem_scores[i] = bm25_meta_scores[i]
        elif method == "tfidf":
            native_sem_scores[i] = tfidf_native_scores[i]
            meta_sem_scores[i] = tfidf_meta_scores[i]
        else:  # hybrid
            native_sem_scores[i] = alpha * bm25_native_scores[i] + (1 - alpha) * tfidf_native_scores[i]
            meta_sem_scores[i] = alpha * bm25_meta_scores[i] + (1 - alpha) * tfidf_meta_scores[i]

    # 4) soft potential + final score
    results: List[Dict[str, Any]] = []

    for i, card in enumerate(dataset_cards):
        # ---------- hard filter: discard incompatible modalities ----------
        if not is_modality_compatible(subtask, card):
            continue

        meta = getattr(card, "meta", {}) or {}
        fields = meta.get("fields", {}) or {}

        native_potential = soft_native_match(fields)
        meta_potential = soft_meta_match(fields, pref)
        meta_richness = compute_meta_richness(fields)  # NEW

        native_sem = native_sem_scores[i]
        meta_sem = meta_sem_scores[i]

        # native view logic unchanged
        native_score = 0.7 * native_sem + 0.3 * native_potential

        # meta view introduces soft boost for meta_richness
        # semantic 0.7 + role match 0.2 + richness 0.1
        meta_score = 0.7 * meta_sem + 0.2 * meta_potential + 0.1 * meta_richness

        # Keep native/meta at 0.45/0.55 so both type-A and type-B datasets can enter
        w_native, w_meta = 0.45, 0.55
        final_score = w_native * native_score + w_meta * meta_score

        results.append({
            "dataset_id": card.dataset_id,
            "dataset_name": card.name,
            "final_score": float(final_score),
            "native_score": float(native_score),
            "meta_score": float(meta_score),
        })


    # 5) sort & take top_k (no strong threshold filtering, ensure recall)
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]

from tools.planner_tools.grounding.dataset_preference_tools import plan_dataset_preference
@register_tool("initial_dataset_retrieve")
def initial_dataset_retrieve(
    subtasks: List[Dict[str, Any]],
    dataset_cards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Initial retrieval over all subtasks (concurrent version)
    """
    print("Starting initial dataset retrieval for subtasks...")
    if not subtasks:
        return []

    def _process_one_subtask(subtask: Dict[str, Any]) -> Dict[str, Any]:
        ds_pref = plan_dataset_preference(subtask)
        retrieval_result = retrieve_candidates(
            subtask,
            dataset_cards,
            ds_pref
        )
        st_aug = dict(subtask)
        st_aug["candidates"] = retrieval_result
        st_aug["dataset_preference"] = ds_pref
        return st_aug

    max_workers = min(6, len(subtasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        subtasks_aug = list(executor.map(_process_one_subtask, subtasks))
    print("Initial dataset retrieval completed.")
    return subtasks_aug