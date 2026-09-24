# ---------------------------
# Logging setup
# ---------------------------
import logging
import os

def setup_logger(results_path):
    logger = logging.getLogger("GA_Run")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    if results_path:
        os.makedirs(results_path, exist_ok=True)
        fh = logging.FileHandler(f"{results_path}/run.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger