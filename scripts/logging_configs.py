import logging


def set_logger_level(logger: logging.Logger, level: int):
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Function to return logger.
    """
    logger = logging.getLogger(name)

    # Only add a handler if none exist
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False  # prevent bubbling to root logger
    else:
        # update existing handler levels too
        for h in logger.handlers:
            h.setLevel(level)

    return logger
