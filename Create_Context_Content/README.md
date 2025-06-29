# Layered Python Application Example

This project demonstrates a layered application structure in Python, including Logger, DatabaseConnection, UserRepository, NotificationService, and UserService classes. It also includes a method tree explorer that outputs the structure in both text and JSON formats, optimized for AI context generation.

## Features
- Layered architecture with clear separation of concerns
- Object graph exploration to output method trees
- AI-friendly context generation in markdown format
- Automatic file naming with class name and timestamp
- Output saved to dedicated folder
- Logging functionality
- Cycle detection to prevent infinite recursion
- Example main block for demonstration

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
