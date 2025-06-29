"""
Main entry point for the layered application example.

This script demonstrates:
1. Creating a layered application structure with proper dependency injection
2. Using the ContextCreator to explore and document the object graph
3. Generating both text and JSON representations of the method tree
"""

from example_objects import Logger, DatabaseConnection, UserRepository, NotificationService, UserService
from context_creator import ContextCreator

if __name__ == "__main__":
    # Instantiate shared components
    logger = Logger()
    db_conn = DatabaseConnection()
    
    # Build service layer
    repo = UserRepository(db=db_conn, logger=logger)
    notifier = NotificationService(logger=logger)
    user_service = UserService(repo=repo, notifier=notifier, logger=logger)
    
    # Create context explorer with default output directory "generated_context"
    context = ContextCreator()
    
    # Reflect and save (filenames will be auto-generated based on class name and timestamp)
    text_file = context.save_method_tree_text(user_service)
    json_file = context.save_method_tree_json(user_service)
    
    print(f"Method tree files generated: {text_file}, {json_file}")
