"""
This module contains example domain objects demonstrating a layered application architecture.
It shows dependency injection patterns and separation of concerns between data, business logic,
and infrastructure components.
"""

class Logger:
    """
    Handles logging of application events and errors.
    
    This is a simple implementation that prints to console, but could be
    extended to write to files, external logging services, etc.
    """
    def log_info(self, message: str):
        """Log an informational message."""
        print(f"INFO: {message}")
        
    def log_error(self, message: str):
        """Log an error message."""
        print(f"ERROR: {message}")

class DatabaseConnection:
    """
    Manages database connections and executes queries.
    
    This is a simplified implementation without actual database interaction.
    In a real application, this would connect to a database system.
    """
    def connect(self):
        """Establish connection to the database."""
        pass
        
    def disconnect(self):
        """Close the database connection."""
        pass
        
    def execute_query(self, query: str) -> list:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string to execute
            
        Returns:
            List of query results (empty in this example)
        """
        return []

class UserRepository:
    """
    Data access layer for user-related operations.
    
    This repository encapsulates all database operations related to users,
    providing a clean API for the service layer.
    """
    def __init__(self, db: DatabaseConnection, logger: Logger):
        """
        Initialize the repository with required dependencies.
        
        Args:
            db: Database connection to use for queries
            logger: Logger for recording operations and errors
        """
        self.db = db
        self.logger = logger
        
    def get_user_by_id(self, user_id: int):
        """
        Retrieve a user by their ID.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            User data (not implemented in this example)
        """
        pass
        
    def save_user(self, user_data: dict):
        """
        Create or update a user in the database.
        
        Args:
            user_data: Dictionary containing user properties
        """
        pass

class NotificationService:
    """
    Service for sending notifications to users through various channels.
    
    This service abstracts the details of different notification methods,
    providing a unified interface for the application.
    """
    def __init__(self, logger: Logger):
        """
        Initialize the notification service.
        
        Args:
            logger: Logger for recording notification attempts and errors
        """
        self.logger = logger
        
    def send_email(self, address: str, content: str):
        """
        Send an email notification.
        
        Args:
            address: Email address of the recipient
            content: Content of the email message
        """
        pass
        
    def send_sms(self, number: str, content: str):
        """
        Send an SMS notification.
        
        Args:
            number: Phone number of the recipient
            content: Content of the SMS message
        """
        pass

class UserService:
    """
    Business logic layer for user-related operations.
    
    This service coordinates between repositories and other services
    to implement business processes related to users.
    """
    def __init__(self, repo: UserRepository, notifier: NotificationService, logger: Logger):
        """
        Initialize the user service with required dependencies.
        
        Args:
            repo: User repository for data access
            notifier: Notification service for sending alerts
            logger: Logger for recording operations and errors
        """
        self.repository = repo
        self.notifier = notifier
        self.logger = logger
        
    def register_user(self, user_data: dict):
        """
        Register a new user in the system.
        
        This would typically:
        1. Validate the user data
        2. Save the user to the database
        3. Send a welcome notification
        
        Args:
            user_data: Dictionary containing user registration information
        """
        pass
        
    def notify_user(self, user_id: int, message: str):
        """
        Send a notification to a specific user.
        
        This would typically:
        1. Look up the user's contact information
        2. Determine the preferred notification channel
        3. Send the notification
        
        Args:
            user_id: ID of the user to notify
            message: Content of the notification
        """
        pass
