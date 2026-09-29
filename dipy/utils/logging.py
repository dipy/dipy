from contextlib import contextmanager
from enum import IntEnum
import logging
import os
import sys

from dipy.utils.deprecator import warning_for_keywords


class CustomHandler(logging.Handler):
    """Custom logging handler that writes an empty line for empty log messages,
    otherwise formats the message as usual.
    """

    @warning_for_keywords(from_version="1.13.0")
    def __init__(self, *, stream=None, filename=None):
        super().__init__()  # call with no arguments
        if filename is not None:
            self._should_close = True
            self.stream = open(filename, "a", encoding="utf-8")
        else:
            self._should_close = False
            self.stream = stream if stream is not None else sys.stdout

    def emit(self, record):
        try:
            msg = record.getMessage()
            if msg == "":
                self.stream.write("\n")
            else:
                formatted = self.format(record)
                self.stream.write(formatted + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()

    def close(self):
        if self._should_close:
            try:
                self.stream.close()
            except Exception:
                pass
        super().close()


@warning_for_keywords(from_version="1.13.0")
def get_logger(*, name="dipy", filename=None, force=False):
    """Return a logger instance configured for DIPY.

    Parameters
    ----------
    name : str
        The logger name.
    filename : str, Path or None, optional
        If provided, log messages will also be saved to this file. If ``None``,
        logs are sent to stdout.
    force : bool, optional
        If True, existing handlers attached to the logger will be removed
        and replaced with a new handler. This allows reconfiguration of the
        logger even if it was already set up.

    Returns
    -------
    _logger : logging.Logger
        Configured logger.
    """

    _logger = logging.getLogger(name)
    if force or not _logger.hasHandlers():
        _logger.handlers.clear()
        if filename:
            handler = CustomHandler(filename=filename)
        else:
            handler = CustomHandler(stream=sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        # Enable propagation when running tests so pytest caplog can capture messages
        # before we were using False to prevent propagation to root logger
        _logger.propagate = "PYTEST_CURRENT_TEST" in os.environ
    return _logger


@warning_for_keywords(from_version="1.13.0")
def configure_logger(
    *,
    level=logging.INFO,
    fmt="[%(asctime)s][%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=None,
):
    """Reconfigure DIPY logger.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).
    fmt : str, optional
        Log message format.
    datefmt : str, optional
        Date format for log messages.
    filename : str, Path or None, optional
        If provided, log messages will also be saved to this file. If ``None``,
        logs are sent to stdout.
    """

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if filename:
        handler = CustomHandler(filename=filename)
    else:
        handler = CustomHandler(stream=sys.stdout)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

    dipy_logger = logging.getLogger("dipy")
    if dipy_logger.hasHandlers():
        dipy_logger.setLevel(level)
        for h in dipy_logger.handlers:
            h.setLevel(level)


def add_file_handler(
    filename,
    *,
    level=None,
    fmt="[%(asctime)s][%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
):
    """Add a file handler to the DIPY logger without removing existing handlers.

    Parameters
    ----------
    filename : str or Path
        Path to the log file. The file is opened in append mode.
    level : int, optional
        Logging level for the file handler. If None, uses the current logger
        level.
    fmt : str, optional
        Log message format.
    datefmt : str, optional
        Date format for log messages.

    Returns
    -------
    handler : CustomHandler
        The handler that was added, to be passed to
        :func:`remove_file_handler` when the file should stop receiving
        messages.
    """
    _logger = logging.getLogger("dipy")
    if level is None:
        level = _logger.getEffectiveLevel()
    handler = CustomHandler(filename=str(filename))
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    _logger.addHandler(handler)
    return handler


def remove_file_handler(handler):
    """Detach a handler previously added by :func:`add_file_handler`.

    Parameters
    ----------
    handler : logging.Handler
        The handler to remove. It is closed after removal. Removing a
        handler that is not attached is a no-op.
    """
    _logger = logging.getLogger("dipy")
    _logger.removeHandler(handler)
    handler.close()


@contextmanager
def file_handler(filename, *, level=None):
    """Send DIPY log messages to ``filename`` for the duration of the block.

    Parameters
    ----------
    filename : str or Path
        Path to the log file. The file is opened in append mode.
    level : int, optional
        Logging level for the file handler. If None, uses the current logger
        level.

    Yields
    ------
    handler : CustomHandler
        The handler writing to ``filename``.
    """
    handler = add_file_handler(filename, level=level)
    try:
        yield handler
    finally:
        remove_file_handler(handler)


# Provide a default logger for convenience
logger = get_logger()


class VerbosityLevels(IntEnum):
    """Verbosity levels for dipy algorithms.

    NONE : do not print anything
    STATUS : print information about the current status of the algorithm
    DIAGNOSE : print high level information of the components involved in the
        registration that can be used to detect a failing component
    DEBUG : print as much information as possible to isolate the cause of a bug
    """

    NONE = 0
    STATUS = 1
    DIAGNOSE = 2
    DEBUG = 3
