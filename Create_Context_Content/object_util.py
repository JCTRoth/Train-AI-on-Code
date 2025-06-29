"""
Utility module for exploring object structures and generating method trees.

Note: This module's functionality has been moved to the ContextCreator class.
It's maintained here for backward compatibility.
"""

import inspect
import json
from typing import Any, Set

def _get_methods(obj: Any):
    """
    Extract and format methods from an object with their signature information.
    
    Args:
        obj: The object to extract methods from
        
    Returns:
        A list of formatted method signatures
    """
    methods = []
    for name, member in inspect.getmembers(obj, predicate=inspect.ismethod):
        if not name.startswith("__"):
            sig = inspect.signature(member)
            params = ", ".join(f"{k}: {v.annotation.__name__ if v.annotation != inspect._empty else 'Any'}" for k, v in sig.parameters.items() if k != 'self')
            methods.append(f".{name}({params})")
    return methods

def _explore_object(obj: Any, name: str, visited: Set[int], indent: int = 0, lines=None, json_tree=None):
    """
    Recursively explore an object and its attributes.
    
    This function builds both text and JSON representations of an object graph,
    tracking visited objects to avoid infinite recursion.
    
    Args:
        obj: The object to explore
        name: The name of the object
        visited: Set of object IDs already visited
        indent: Current indentation level
        lines: Accumulated text lines
        json_tree: Accumulated JSON structure
        
    Returns:
        Tuple of (text lines, JSON tree)
    """
    if lines is None:
        lines = []
    if json_tree is None:
        json_tree = {}
    obj_id = id(obj)
    prefix = "  " * indent
    class_name = obj.__class__.__name__
    lines.append(f"{prefix}{name} -> {class_name}")
    json_tree['name'] = name
    json_tree['class'] = class_name
    json_tree['methods'] = _get_methods(obj)
    json_tree['children'] = []
    if obj_id in visited:
        lines.append(f"{prefix}  (Already visited)")
        return lines, json_tree
    visited.add(obj_id)
    for attr, value in vars(obj).items():
        if hasattr(value, "__class__") and not isinstance(value, (str, int, float, list, dict, set, tuple, type(None))):
            child_json = {}
            child_lines, child_json = _explore_object(value, attr, visited, indent + 1, [], child_json)
            lines.extend(child_lines)
            json_tree['children'].append(child_json)
    return lines, json_tree

def save_method_tree_text(obj: Any, filename: str = "method_tree.txt"):
    """
    Create and save a text representation of an object's method tree.
    
    Args:
        obj: The root object to explore
        filename: Output filename for the text representation
    """
    lines, _ = _explore_object(obj, "root", set())
    with open(filename, "w") as f:
        for line in lines:
            f.write(line + "\n")

def save_method_tree_json(obj: Any, filename: str = "method_tree.json"):
    """
    Create and save a JSON representation of an object's method tree.
    
    Args:
        obj: The root object to explore
        filename: Output filename for the JSON representation
    """
    _, json_tree = _explore_object(obj, "root", set())
    with open(filename, "w") as f:
        json.dump(json_tree, f, indent=2)
