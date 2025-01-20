import logging
import os
import sys
import re

class StreamToLogger:
    """
    Redirects writes to a logger instance.
    """
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        if message.strip():  # Avoid logging empty messages
            self.logger.log(self.level, message.strip())

    def flush(self) -> None:
        pass  # Required for compatibility with `sys.stdout`

def setup_logger(log_dir: str = "logs", level: int = logging.INFO, log_note: str = None) -> str:
    """
    Sets up a logger that writes to a uniquely indexed file inside a logs directory.
    """

    os.makedirs(log_dir, exist_ok=True)

    # Regex to extract indices from filenames: 'log_<note>_<index>.log' or 'log_<index>.log'
    log_pattern = re.compile(r"log(?:_[a-zA-Z0-9_]+)?_(\d+)\.log")

    # Extract indices from existing log files
    existing_logs = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    indices = [dataset
        int(log_pattern.search(f).group(1))
        for f in existing_logs
        if log_pattern.search(f)
    ]

    # Next index
    next_index = max(indices, default=-1) + 1

    # Create the new log file path with optional note
    if log_note:
        log_file = os.path.join(log_dir, f"log_{log_note}_{next_index}.log")
    else:
        log_file = os.path.join(log_dir, f"log_{next_index}.log")

    # Configure the logger
    logging.basicConfig(
        level=level,
        format='[ %(asctime)s ]  %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Write logs to the file
            logging.StreamHandler(sys.stdout),  # Also log to the console
        ]
    )

    # Redirect stdout and stderr to the logger
    logger = logging.getLogger()
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return log_file