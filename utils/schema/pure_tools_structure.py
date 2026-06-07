from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

@dataclass
class PureToolSpec:
    name: str
    category: str
    description: str
    io_inputs: List[str]
    io_outputs: List[str]
    behavior: List[str]
    params_doc: Dict[str, str]           
    typical_uses: List[str]

    param_schema: Dict[str, Any]         # description of the structure of each parameter
    return_schema: Dict[str, Any]        # description of the structure of the tool's return value

    backend: Callable[..., Dict[str, Any]]

    raw_cfg: Dict[str, Any]



