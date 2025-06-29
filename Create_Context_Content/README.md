# Layered Python Application Example

This project demonstrates a layered application structure in Python, including Logger, DatabaseConnection, UserRepository, NotificationService, and UserService classes. It also includes a method tree explorer that outputs the structure in both text and JSON formats.

## Features
- Layered architecture with clear separation of concerns
- Object graph exploration to output method trees
- Automatic file naming with class name and timestamp
- Output saved to dedicated folder
- Logging functionality
- Example main block for demonstration

## Usage
Run the main script:

```fish
python3 main.py
```

This will generate files like `userservice_1751235868.txt` and `userservice_1751235868.json` in the `generated_context` directory. The files are named after the object class with a timestamp appended.
