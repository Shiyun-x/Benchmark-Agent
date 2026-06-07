"""
TopicMetadata
    ├─ topic / area / global_modalities
    ├─ subtasks                ← produced by AnalystAgent
    ├─ quotas                  ← produced by PlannerAgent
    ├─ transform_mix           ← produced by PlannerAgent
    ├─ candidate_datasets      ← produced by Retriever + Scorer
    ├─ selected_samples        ← produced by Cleaner
    └─ benchmark_items         ← produced by Benchmark Generator
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    """Base class for all domain-level metadata."""

    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
    )


# -----------------------------
# SubTasks (Analyst Output)
# -----------------------------


class SampleSchema(DomainModel):
    """Schema defining input/output fields for samples in this subtask."""

    input: Dict[str, Any] = Field(..., description="Input fields schema.")
    output: Dict[str, Any] = Field(..., description="Output fields schema.")


class SubTask(DomainModel):
    """A decomposed capability/task derived from the input topic."""

    id: str = Field(..., description="Machine-readable task ID.")
    name: Optional[str] = Field(None, description="Human-readable name for the task.")
    description: str = Field(..., description="Natural language description of the subtask.")
    task: Optional[str] = Field(
        None,
        description="High-level task label (e.g., 'classification_multiclass', 'retrieval').",
    )
    capability_target: Optional[str] = Field(
        None,
        description=(
            "Primary capability targeted by this subtask "
            "(e.g., 'perception', 'understanding', 'reasoning', 'memory')."
        ),
    )
    domain: Optional[str] = Field(
        None,
        description="Higher-level domain (e.g., 'video_understanding').",
    )
    robustness_requirement: Optional[bool] = Field(
        None,
        description="Whether robustness is a requirement for this subtask.",
    )
    sample_schema: SampleSchema = Field(
        ...,
        description="Schema defining input/output fields for samples in this subtask.",
    )

    retrieval_result: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Initial retrieval results (dataset_id + score) from Retriever.",
    )
    scored_candidates: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scored candidate datasets with detailed scoring breakdowns.",
    )

    importance: Optional[float] = Field(
        None,
        description="Planned importance score for this subtask.",
    )
    quota_plan: Optional[int] = Field(
        None,
        description="Planned quota allocation for this subtask.",
    )


# -----------------------------
# Quotas, Difficulty, Robustness (Planner Output)
# -----------------------------


class SubTaskQuota(DomainModel):
    """Sample quota planned for each subtask, including difficulty and priority."""

    subtask_id: str = Field(..., description="Reference to SubTask.id.")
    target_samples: int = Field(
        ...,
        description="Target number of benchmark items for this subtask.",
    )
    difficulty_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribution over difficulty levels (proportions, not counts).",
    )
    priority: Optional[float] = Field(
        None,
        description="Priority for this subtask, used for downstream allocation order.",
    )


class TransformMix(DomainModel):
    """
    Transformation mix defining the percentage of augmentation / rewriting /
    corruption operations for a given subtask.
    """

    subtask_id: str = Field(..., description="Reference to SubTask.id.")
    operations: Dict[str, float] = Field(
        default_factory=dict,
        description="Operation name → percentage (values typically sum to ~1.0).",
    )


# -----------------------------
# Dataset Pool & Candidate Datasets
# -----------------------------


class DatasetCandidate(DomainModel):
    """A candidate dataset selected from the dataset pool."""

    subtask_id: str = Field(..., description="ID of the subtask this candidate is matched to.")
    dataset_id: str = Field(..., description="Internal dataset ID.")
    name: str = Field(..., description="Dataset name (e.g., COCO, MSRVTT).")
    source: Optional[str] = Field(
        None,
        description="Dataset source path/URL/identifier.",
    )
    modalities: List[str] = Field(
        default_factory=list,
        description="Supported modalities by this dataset.",
    )
    labels: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-level tags (e.g., annotation type, domain).",
    )

    retrieval_score: Optional[float] = Field(
        None,
        description="Score from BM25/TF-IDF retrieval stage (semantic or lexical match).",
    )
    retrieval_detail: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed retrieval metrics, e.g., {'bm25': ..., 'tfidf': ..., 'hybrid': ...}.",
    )
    retrieval_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Non-numeric retrieval metadata, e.g. {'method': 'hybrid', 'alpha': 0.6}.",
    )
    score: Optional[float] = Field(
        None,
        description="Final overall relevance score (from LLM scoring stage).",
    )
    score_detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fine-grained scoring breakdown from LLM or heuristic criteria.",
    )


# -----------------------------
# Sample Selection (Cleaner Output)
# -----------------------------


class SampleSpec(DomainModel):
    """
    A selected sample from a dataset, before generating the final benchmark item.

    Cleaners produce SampleSpecs; Benchmark Generator later expands them into
    executable benchmark instances.
    """

    dataset_id: str = Field(..., description="Reference to DatasetCandidate.dataset_id.")
    sample_id: str = Field(..., description="Dataset-internal sample reference.")
    subtask_id: str = Field(..., description="SubTask this sample is used for.")
    difficulty: Optional[str] = Field(
        None,
        description="Difficulty level assigned to this sample.",
    )
    robustness_tags: List[str] = Field(
        default_factory=list,
        description="Robustness slice labels (e.g., 'night', 'occlusion').",
    )
    applied_transforms: List[str] = Field(
        default_factory=list,
        description="List of applied transformation/augmentation operations.",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (duration, fps, bbox stats, etc.).",
    )


# -----------------------------
# Final Benchmark Item (Generator Output)
# -----------------------------


class BenchmarkItem(DomainModel):
    """Final benchmark instance used in evaluation."""

    id: str = Field(..., description="Unique benchmark item ID.")
    subtask_id: str = Field(..., description="Reference to SubTask.id.")
    input: Dict[str, Any] = Field(
        ...,
        description="Structured multi-modal input (e.g., video path + text query).",
    )
    output: Dict[str, Any] = Field(
        ...,
        description="Expected output structure matching SubTask.io.",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (difficulty, origin dataset, transforms, etc.).",
    )


class TopicMetadata(DomainModel):
    """
    The global state object shared across all agents in the benchmark-building workflow.

    Each agent reads/writes part of this structure.
    """

    task_id: str = Field(..., description="Machine-readable task ID.")
    target_size: int = Field(..., description="Number of samples to generate.")
    description: Optional[str] = Field(
        None,
        description="The main description provided by the user.",
    )

    topic: Optional[str] = Field(None, description="")
    area: Optional[str] = Field(
        None,
        description="Broader category (e.g., 'medical_imaging', 'video_understanding').",
    )
    global_modalities: List[str] = Field(
        default_factory=list,
        description="All modalities involved in this benchmark.",
    )

    keywords: List[str] = Field(
        default_factory=list,
        description="A list of keywords that describe the task.",
    )
    short_topic: Optional[str] = Field(
        None,
        description="The main description provided by the user.",
    )
    modalities: List[str] = Field(
        default_factory=list,
        description="All modalities involved in this benchmark.",
    )
    subtasks: List[SubTask] = Field(
        default_factory=list,
        description="List of decomposed subtasks.",
    )

    allocation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall allocation plan produced by the Planner Agent.",
    )

    candidate_datasets: Dict[str, List[DatasetCandidate]] = Field(
        default_factory=dict,
        description="Candidate datasets found from the dataset pool.",
    )
    selected_datasets: List[DatasetCandidate] = Field(
        default_factory=list,
        description="Filtered & cleaned dataset-level selections.",
    )
    selected_samples: List[SampleSpec] = Field(
        default_factory=list,
        description="Filtered & cleaned sample-level selections.",
    )

    benchmark_items: List[BenchmarkItem] = Field(
        default_factory=list,
        description="Final benchmark items ready for evaluation.",
    )


__all__ = [
    "DomainModel",
    "SampleSchema",
    "SubTask",
    "SubTaskQuota",
    "TransformMix",
    "DatasetCandidate",
    "SampleSpec",
    "BenchmarkItem",
    "TopicMetadata",
]
