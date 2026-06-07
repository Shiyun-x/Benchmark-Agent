# benchmark_agent/domain/dataset_card.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class IOSchema(BaseModel):
    in_: Optional[List[Any]] = Field(default=None, alias="in")
    out: Optional[List[Any]] = None

    # Pydantic v2 config (replaces `allow_population_by_field_name=True`)
    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_name=True,
        extra="allow",
    )


class DatasetCard(BaseModel):
    """
    - native layer: description / card_text / tasks / domain
    - meta: directly store the dict returned by LLM, without field-level strong validation
    - raw_meta: source_json, old tags, etc.
    """

    # Basic information
    dataset_id: str
    name: str

    modalities: Optional[List[str]] = None
    io_schemas: Optional[List] = None
    size_samples: Optional[int] = None

    # native layer
    description: str = ""
    card_text: str = ""
    tasks: List[str] = Field(default_factory=list)
    domain: str = ""

    # meta layer
    meta: Dict[str, Any] = Field(default_factory=dict)

    raw_meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


def create_dataset_card_from_raw(raw: Dict[str, Any]) -> DatasetCard:
    """
    Create a DatasetCard instance from a raw dict.
    """
    return DatasetCard(**raw)
