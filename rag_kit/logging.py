import logging
from functools import wraps
from time import time
from typing import Any, Callable


def setup_logger(name: str, level=logging.INFO):
    """Sets up a logger with the specified name."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger = logging.LoggerAdapter(logger)
    return logger


def alog_timing(logger: logging.Logger):
    def adecorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Async logger decorator to log the execution of async functions.

        Args:
            func (Callable): The async function to wrap.

        Returns:
            Callable: The wrapped async function with logging.
        """

        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time()
            logging.info(f"Executing {func_name}")
            result = await func(*args, **kwargs)
            elapsed_time = time() - start_time
            elapsed_time_ms = elapsed_time * 1000
            logger.info(
                f"Executed {func_name} in {elapsed_time:.2f}s ({elapsed_time_ms:.0f}ms)"
            )
            return result

        return wrapper

    return adecorator


def log_timing(logger: logging.Logger):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Sync logger decorator to log the execution of synchronous functions.

        Args:
            func (Callable): The function to wrap.

        Returns:
            Callable: The wrapped function with logging.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time()
            logging.info(f"Executing {func_name}")
            result = func(*args, **kwargs)
            elapsed_time = time() - start_time
            elapsed_time_ms = elapsed_time * 1000
            logger.info(
                f"Executed {func_name} in {elapsed_time:.2f}s ({elapsed_time_ms:.0f}ms)"
            )
            return result

        return wrapper

    return decorator
