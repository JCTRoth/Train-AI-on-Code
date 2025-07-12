import inspect
import json
import os
import time
import logging
from typing import Any, Dict, List, Set, Tuple

# Configure logging for terminal output only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ContextCreator:
    """
    A class for exploring and documenting object structures and their method trees.
    Creates visual representations of object relationships and methods for documentation purposes.
    """
    
    def __init__(self, output_dir: str = "generated_context"):
        """
        Initialize the ContextCreator with an output directory.
        
        Args:
            output_dir: Directory where generated files will be stored
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
    
    @staticmethod
    def _get_methods(obj: Any) -> List[Dict[str, Any]]:
        """
        Extract methods from an object with their signature information.
        
        Args:
            obj: The object to extract methods from
            
        Returns:
            A list of dictionaries containing method information
        """
        methods = []
        for name, member in inspect.getmembers(obj, predicate=inspect.ismethod):
            if not name.startswith("__"):
                try:
                    sig = inspect.signature(member)

                    # Get parameters information
                    params = []
                    for k, v in sig.parameters.items():
                        if k != 'self':
                            param_type = v.annotation.__name__ if v.annotation != inspect._empty else 'Any'
                            default_value = "" if v.default == inspect._empty else f"={v.default}"
                            params.append(f"{k}: {param_type}{default_value}")

                    # Get return type information
                    return_type = sig.return_annotation.__name__ if sig.return_annotation != inspect._empty else 'Any'

                    # Get docstring if available
                    doc = inspect.getdoc(member)
                    brief_doc = doc.split('\n')[0] if doc else "No description available"

                    # Create method info dictionary
                    method_info = {
                        'name': name,
                        'signature': f".{name}({', '.join(params)})",
                        'params': params,
                        'return_type': return_type,
                        'doc': brief_doc
                    }

                    methods.append(method_info)
                except Exception as e:
                    # Fallback for methods that can't be inspected
                    methods.append({
                        'name': name,
                        'signature': f".{name}(...)",
                        'params': [],
                        'return_type': 'Unknown',
                        'doc': "Could not inspect method"
                    })

        return methods

    @classmethod
    def _explore_object(cls, obj: Any, name: str, visited: Set[int], 
                      indent: int = 0, lines: List[str] = None, 
                      json_tree: Dict[str, Any] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Recursively explore an object and its attributes.
        
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
        
        # Add to text representation
        lines.append(f"{prefix}{name} -> {class_name}")
        
        # Add to JSON representation
        json_tree['name'] = name
        json_tree['class'] = class_name
        json_tree['methods'] = cls._get_methods(obj)
        json_tree['children'] = []
        
        # Check for cycles
        if obj_id in visited:
            lines.append(f"{prefix}  (Already visited)")
            return lines, json_tree
            
        visited.add(obj_id)
        
        # Explore attributes
        for attr, value in vars(obj).items():
            if (hasattr(value, "__class__") and 
                not isinstance(value, (str, int, float, list, dict, set, tuple, type(None)))):
                child_json = {}
                child_lines, child_json = cls._explore_object(
                    value, attr, visited, indent + 1, [], child_json
                )
                lines.extend(child_lines)
                json_tree['children'].append(child_json)
                
        return lines, json_tree

    def save_method_tree_text(self, obj: Any, filename: str = None) -> str:
        """
        Create and save a text representation of an object's method tree.
        
        Args:
            obj: The root object to explore
            filename: Optional output filename for the text representation.
                      If None, a name will be generated based on the object class and timestamp.
        
        Returns:
            The filename of the saved file
        """
        # Generate filename if not provided
        if filename is None:
            unix_time = int(time.time())
            class_name = obj.__class__.__name__.lower()
            filename = f"{class_name}_{unix_time}.txt"
        
        # Ensure file is saved in the output directory
        filepath = os.path.join(self.output_dir, filename)
        
        # Get raw tree data
        lines, json_tree = self._explore_object(obj, "root", set())
        
        # Generate enhanced AI-friendly text representation
        ai_context_lines = self._generate_ai_context(json_tree)
        
        # Write to file
        with open(filepath, "w") as f:
            f.write(ai_context_lines)
        
        self.logger.info(f"Saved enhanced text representation to {filepath}")
        return filepath
        
    def _generate_ai_context(self, json_tree: Dict[str, Any], depth: int = 0) -> str:
        """
        Generate AI-friendly context text from the JSON tree.
        
        Args:
            json_tree: The JSON tree structure
            depth: Current depth in the tree
            
        Returns:
            Formatted text optimized for AI context
        """
        lines = []
        indent = "  " * depth
        
        # Add header for the component
        if depth == 0:
            lines.append(f"# {json_tree['class']} Component Structure")
            lines.append("")
            lines.append(f"Component analysis of {json_tree['class']} object hierarchy and capabilities.")
            lines.append("")
        else:
            lines.append(f"{indent}## {json_tree['name']}: {json_tree['class']}")
            
        # Add method documentation with improved details
        if json_tree['methods']:
            if depth == 0:
                lines.append("## Available Methods")
            else:
                lines.append(f"{indent}### Methods")
                
            for method in json_tree['methods']:
                method_name = method['name']
                return_type = method['return_type']

                # Format parameters for display
                if method['params']:
                    param_str = ", ".join(method['params'])
                    lines.append(f"{indent}- `{method_name}({param_str}) -> {return_type}`")

                    # Add method description if available
                    if method['doc'] and method['doc'] != "No description available":
                        lines.append(f"{indent}  {method['doc']}")
                else:
                    lines.append(f"{indent}- `{method_name}() -> {return_type}`")

                    # Add method description if available
                    if method['doc'] and method['doc'] != "No description available":
                        lines.append(f"{indent}  {method['doc']}")

            lines.append("")
        
        # Add dependencies section with better organization
        if json_tree['children']:
            if depth == 0:
                lines.append("## Dependencies")
                lines.append("")
            else:
                lines.append(f"{indent}### Dependencies")
                lines.append("")
            
            for child in json_tree['children']:
                child_text = self._generate_ai_context(child, depth + 1)
                lines.append(child_text)
                
        # For root level, add a concise and informative summary
        if depth == 0:
            total_methods = len(json_tree['methods'])
            direct_deps = len(json_tree['children'])

            # Count all methods in the tree
            all_methods = total_methods
            for child in json_tree['children']:
                all_methods += self._count_methods_recursive(child)

            lines.append("## Overview")
            lines.append("")
            lines.append(f"- **Type**: {json_tree['class']}")
            lines.append(f"- **Direct Methods**: {total_methods}")
            lines.append(f"- **Total Methods** (including dependencies): {all_methods}")
            lines.append(f"- **Direct Dependencies**: {direct_deps}")
            lines.append("")

        return "\n".join(lines)

    def _count_methods_recursive(self, json_tree: Dict[str, Any]) -> int:
        """
        Count the total number of methods in a tree recursively.

        Args:
            json_tree: The JSON tree to count methods in

        Returns:
            Total number of methods
        """
        count = len(json_tree['methods'])
        for child in json_tree['children']:
            count += self._count_methods_recursive(child)
        return count

    def save_method_tree_json(self, obj: Any, filename: str = None) -> str:
        """
        Create and save a JSON representation of an object's method tree.
        
        Args:
            obj: The root object to explore
            filename: Optional output filename for the JSON representation.
                      If None, a name will be generated based on the object class and timestamp.
        
        Returns:
            The filename of the saved file
        """
        # Generate filename if not provided
        if filename is None:
            unix_time = int(time.time())
            class_name = obj.__class__.__name__.lower()
            filename = f"{class_name}_{unix_time}.json"
        
        # Ensure file is saved in the output directory
        filepath = os.path.join(self.output_dir, filename)
            
        _, json_tree = self._explore_object(obj, "root", set())
        with open(filepath, "w") as f:
            json.dump(json_tree, f, indent=2)
        
        self.logger.info(f"Saved JSON representation to {filepath}")
        return filepath
