from typing import Callable, Dict, Any, Union, Literal, List, Optional
from dataclasses import dataclass, asdict
import inspect
@dataclass
class FunctionInfo:
    name: str
    func: Callable
    args: List[str]
    docstring: Optional[str]
    body: str
    return_type: Optional[str]
    def to_dict(self) -> dict:
        # using asdict, but exclude func field because it cannot be serialized
        d = asdict(self)
        d.pop('func')  # remove func field
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FunctionInfo':
        # if you need to create an object from a dictionary
        if 'func' not in data:
            data['func'] = None  # or other default value
        return cls(**data)
class Registry:
    _instance = None
    _registry: Dict[str, Dict[str, Callable]] = {
        "tools": {},
        "agents": {}
    }
    _registry_info: Dict[str, Dict[str, FunctionInfo]] = {
        "tools": {},
        "agents": {}
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, 
                type: Literal["tool", "agent"],
                name: str = None):
        """
        Unified registration decorator.
        Args:
            type: Registration type, "tool" or "agent"
            name: Optional registration name
        """
        def decorator(func: Callable):
            nonlocal name
            if name is None:
                name = func.__name__
                # if type == "agent" and name.startswith('get_'):
                #     name = name[4:]  # strip 'get_' prefix for agents
            
            # collect function metadata
            signature = inspect.signature(func)
            args = list(signature.parameters.keys())
            docstring = inspect.getdoc(func)
            
            # extract function body
            source_lines = inspect.getsource(func)
            # strip decorator and function definition lines
            body_lines = source_lines.split('\n')[1:]  # skip decorator line
            while body_lines and (body_lines[0].strip().startswith('@') or 'def ' in body_lines[0]):
                body_lines = body_lines[1:]
            body = '\n'.join(body_lines)
            
            # read return type hint
            return_type = None
            if signature.return_annotation != inspect.Signature.empty:
                return_type = str(signature.return_annotation)
            
            # build FunctionInfo object
            func_info = FunctionInfo(
                name=name,
                func=func,
                args=args,
                docstring=docstring,
                body=body,
                return_type=return_type
            )
            
            registry_type = f"{type}s"
            self._registry[registry_type][name] = func
            self._registry_info[registry_type][name] = func_info
            return func
        return decorator
    
    @property
    def tools(self) -> Dict[str, Callable]:
        return self._registry["tools"]
    
    @property
    def agents(self) -> Dict[str, Callable]:
        return self._registry["agents"]
    
    @property
    def tools_info(self) -> Dict[str, FunctionInfo]: 
        return self._registry_info["tools"]
    
    @property
    def agents_info(self) -> Dict[str, FunctionInfo]: 
        return self._registry_info["agents"]

# global registry instance
registry = Registry()

# convenience registration helpers
def register_tool(name: str = None):
    return registry.register(type="tool", name=name)

def register_agent(name: str = None):
    return registry.register(type="agent", name=name)

