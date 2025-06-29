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
    def _get_methods(obj: Any) -> List[str]:
        """
        Extract methods from an object with their signature information.
        
        Args:
            obj: The object to extract methods from
            
        Returns:
            A list of formatted method signatures
        """
        methods = []
        for name, member in inspect.getmembers(obj, predicate=inspect.ismethod):
            if not name.startswith("__"):
                sig = inspect.signature(member)
                params = ", ".join(f"{k}: {v.annotation.__name__ if v.annotation != inspect._empty else 'Any'}" 
                                for k, v in sig.parameters.items() if k != 'self')
                methods.append(f".{name}({params})")
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
            lines.append("This document describes the structure and capabilities of a software component.")
            lines.append("")
        else:
            lines.append(f"{indent}## {json_tree['name']}: {json_tree['class']}")
            
        # Add method documentation
        if json_tree['methods']:
            if depth == 0:
                lines.append("## Available Methods")
            else:
                lines.append(f"{indent}### Methods")
                
            for method in json_tree['methods']:
                # Extract method name and parameters from format: .method_name(param1: type, param2: type)
                method_parts = method.split('(', 1)
                method_name = method_parts[0].strip('.')
                params = method_parts[1].strip(')') if len(method_parts) > 1 else ""
                
                if params:
                    lines.append(f"{indent}- `{method_name}`: Takes parameters ({params})")
                else:
                    lines.append(f"{indent}- `{method_name}`: No parameters required")
            
            lines.append("")
        
        # Add dependencies section
        if json_tree['children']:
            if depth == 0:
                lines.append("## Dependencies")
                lines.append("")
                lines.append("This component depends on the following components:")
                lines.append("")
            else:
                lines.append(f"{indent}### Dependencies")
                lines.append("")
            
            for child in json_tree['children']:
                child_text = self._generate_ai_context(child, depth + 1)
                lines.append(child_text)
                
        # For root level, add a summary
        if depth == 0:
            lines.append("## Summary")
            lines.append("")
            lines.append(f"The {json_tree['class']} component has {len(json_tree['methods'])} methods and "
                        f"{len(json_tree['children'])} direct dependencies.")
            lines.append("")
            lines.append("This context was generated automatically by ContextCreator.")
            
        return "\n".join(lines)

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
