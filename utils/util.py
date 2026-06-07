import inspect
import json
import os
from typing import Callable, List, Dict, Any, Optional, Union, get_args, get_origin
from dataclasses import is_dataclass, fields, MISSING
from pydantic import BaseModel
import inquirer


def single_select_menu(options, message: str = ""):
    env_choice = os.getenv("BENCHMARK_CACHE_CHOICE", "").strip()
    if env_choice:
        normalized = {str(opt).strip().lower(): opt for opt in options}
        picked = normalized.get(env_choice.lower())
        if picked is not None:
            print(f"[single_select_menu] auto-select via BENCHMARK_CACHE_CHOICE={picked}")
            return picked

    questions = [
        inquirer.List(
            'choice',
            message=message,
            choices=options,
        ),
    ]
    answers = inquirer.prompt(questions)
    return answers['choice']


def get_type_info(annotation, base_type_map):
    if annotation in base_type_map:
        return {"type": base_type_map[annotation]}

    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)

        if origin is list or origin is List:
            item_type = args[0]
            return {
                "type": "array",
                "items": get_type_info(item_type, base_type_map)
            }

        elif origin is dict or origin is Dict:
            key_type, value_type = args
            if key_type != str:
                raise ValueError("Dictionary keys must be strings")
            if (hasattr(value_type, "__annotations__") or
                    (isinstance(value_type, type) and issubclass(value_type, BaseModel))):
                return get_type_info(value_type, base_type_map)
            return {
                "type": "object",
                "additionalProperties": get_type_info(value_type, base_type_map)
            }

        elif origin is Union:
            types = [get_type_info(arg, base_type_map) for arg in args if arg != type(None)]
            if len(types) == 1:
                return types[0]
            return {"oneOf": types}

    if isinstance(annotation, type):
        try:
            if issubclass(annotation, BaseModel):
                schema = annotation.model_json_schema()
                return {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                    "additionalProperties": False
                }
        except TypeError:
            pass

    if is_dataclass(annotation):
        properties = {}
        required = []
        for field in fields(annotation):
            properties[field.name] = get_type_info(field.type, base_type_map)
            if field.default == field.default_factory == MISSING:
                required.append(field.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

    if hasattr(annotation, "__annotations__"):
        properties = {}
        required = getattr(annotation, "__required_keys__", annotation.__annotations__.keys())
        for key, field_type in annotation.__annotations__.items():
            properties[key] = get_type_info(field_type, base_type_map)
        return {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False
        }

    return {"type": "string"}


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            parameters[param.name] = get_type_info(param.annotation, type_map)
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
