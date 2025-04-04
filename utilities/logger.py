from aiologger import Logger
from aiologger.handlers.files import AsyncFileHandler
from settings import LOG_FILE, LOG_LEVEL

class AsyncLoggerSetup:
    def __init__(self, name: str):
        self.name = name
        self.logger = Logger(name=name, level=LOG_LEVEL)
        self._configure_logger()

    def _configure_logger(self):
        """Configure the asynchronous logger with a file handler."""
        try:
            # Create an asynchronous file handler
            file_handler = AsyncFileHandler(filename=LOG_FILE, mode="a")
            # Add the file handler to the logger (this is an async call; scheduling it here)
            self.logger.add_handler(file_handler)
        except Exception as e:
            # In async contexts you might want to use print if the logger isn't set up yet
            print(f"Error setting up async file handler: {e}")

    def get_logger(self) -> Logger:
        """Return the configured async logger."""
        return self.logger



# Initialize the async logger using the OOP AsyncLoggerSetup class
async_logger_setup = AsyncLoggerSetup("Kubera")
LOGGER = async_logger_setup.get_logger()
