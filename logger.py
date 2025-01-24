import logging

class Logger:
    @staticmethod
    def setup_logger():
        """
        Set up the logger with both file and console handlers.
        """
        logger = logging.getLogger('DataProcessingLogger')
        logger.setLevel(logging.DEBUG)
        
        # File handler (to log into a file)
        file_handler = logging.FileHandler('data_processing.log')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Console handler (to log to the terminal)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Only INFO level messages will show in the terminal
        console_handler.setFormatter(file_format)
        
        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
