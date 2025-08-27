import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('design_requests_bot.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

# Function to get logger for specific module
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)