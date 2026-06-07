"""
Model Configuration Loader
Centralized configuration for all agent and tool models.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration path
DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parent / "resources" / "models.yaml"
)

# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
    
    Returns:
        Dictionary with 'agents' and 'tools' keys.
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model config file not found: {config_path}\n"
            f"Please create the config file or specify a valid path."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Validate structure
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary")
    
    # Set defaults if missing
    if "agents" not in config:
        config["agents"] = {}
    if "tools" not in config:
        config["tools"] = {}
    if "api" not in config:
        config["api"] = {}
    
    # Set default values for agents
    agents_defaults = {
        "analyst": "gpt-5-mini",
        "planner": "gpt-5-mini",
        "generator": "gpt-5-mini",
        "design": "gpt-5-mini",
        "grounding": "gpt-5-mini",
        "allocation": "gpt-5-mini",
    }
    for key, default in agents_defaults.items():
        if key not in config["agents"]:
            config["agents"][key] = default
    
    # Set default values for tools
    tools_defaults = {
        "parse_subtasks": "gpt-5.1",
        "analyst_model": "gpt-5.1",
        "transformability_check": "gpt-5.1",
        "scoring": "gpt-5.1",
        "importance_and_quota": "gpt-5-mini",
        "diagnose_allocation": "gpt-5.1",
        "transform_stage1": "gpt-5.1",
        "transform_stage2": "gpt-5.1",
        "verify": "gpt-5.1",
        "default": "gpt-5.1",
    }
    for key, default in tools_defaults.items():
        if key not in config["tools"]:
            config["tools"][key] = default
    
    _config_cache = config
    return config


def get_agent_model(agent_name: str, config_path: Optional[str] = None) -> str:
    """
    Get model for a specific agent.
    
    Args:
        agent_name: Name of the agent ('analyst', 'planner', 'generator')
        config_path: Optional path to config file
    
    Returns:
        Model name string
    """
    config = load_model_config(config_path)
    return config["agents"].get(agent_name, "gpt-5-mini")


def get_tool_model(tool_name: str, config_path: Optional[str] = None) -> str:
    """
    Get model for a specific tool.
    
    Args:
        tool_name: Name of the tool (see models.yaml for available names)
        config_path: Optional path to config file
    
    Returns:
        Model name string
    """
    config = load_model_config(config_path)
    return config["tools"].get(tool_name, config["tools"].get("default", "gpt-5.1"))


def get_design_model(config_path: Optional[str] = None) -> str:
    """Get Design Agent model."""
    return get_agent_model("design", config_path)


def get_grounding_model(config_path: Optional[str] = None) -> str:
    """Get Grounding Agent model."""
    return get_agent_model("grounding", config_path)


def get_allocation_model(config_path: Optional[str] = None) -> str:
    """Get Allocation Agent model."""
    return get_agent_model("allocation", config_path)


def get_api_key(config_path: Optional[str] = None) -> str:
    """
    Get API key from config file, else from environment LLM_API_KEY.
    """
    config = load_model_config(config_path)
    api = config.get("api") or {}
    key = api.get("api_key") or os.getenv("LLM_API_KEY", "")
    return (key if isinstance(key, str) else "") or ""


def get_api_base_url(config_path: Optional[str] = None) -> str:
    """
    Get API base URL from config file, else from environment LLM_API_BASE_URL.
    """
    config = load_model_config(config_path)
    api = config.get("api") or {}
    url = api.get("base_url")
    return (url if isinstance(url, str) else "")
