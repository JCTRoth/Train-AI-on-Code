import logging


def get_logger():
    return logging


def config_logger():
    formatString = '[%(levelname)s] - %(asctime)s - %(funcName)s - %(message)s'

    formatter = logging.Formatter(formatString)

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format=formatString)

    # Create a file handler - DEBUG
    file_handler = logging.FileHandler(filename='conversion.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler - INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the root logger
    logging.getLogger('').addHandler(file_handler)
    logging.getLogger('').addHandler(console_handler)