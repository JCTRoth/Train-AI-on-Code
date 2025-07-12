# Layered Python Application Example

The Context Creator is a Python application designed
to generate structured context files of class objects it iterates over. It is particularly useful for creating documentation and context for AI code assistants, such as Copilot, by extracting and formatting information about Python classes, methods, and their dependencies.
That way it extracts information from Python classes, methods, and their dependencies, producing Context Files that are used to enrich the available context for AI code assistants. (Copilot)

## Generated Outputs

The application generates two types of outputs:

### Text Output (AI-Friendly Context)
The text output is formatted as markdown and includes:
- Component structure with proper hierarchical headings
- Methods with parameter information
- Dependency relationships
- Summary statistics

Example:
```markdown
# UserService Component Structure

This document describes the structure and capabilities of a software component.

## Available Methods
- `notify_user`: Takes parameters (user_id: int, message: str)
- `register_user`: Takes parameters (user_data: dict)

## Dependencies
...
```

### JSON Output
A structured JSON representation with:
- Component names and class types
- Method signatures
- Hierarchical dependency tree

## Usage
Run the main script:

```fish
python3 main.py
```

This will generate files like `userservice_1751237356.txt` and `userservice_1751237356.json` in the `generated_context` directory. The files are named after the object class with a timestamp appended.

## Code Organization
- `main.py` - Example demonstration
- `example_objects.py` - Layered application classes
- `context_creator.py` - Context generation engine, this class you copy to your porject in order to create the context files.
- `object_util.py` - Legacy utility functions (for backward compatibility)

## Use Cases
This tool is particularly useful for:
- Documenting complex object hierarchies
- Generating context for AI code assistants
- Understanding dependency relationships in large codebases
- Exploring unfamiliar code structures
