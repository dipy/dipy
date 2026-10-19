import io
import logging

import numpy.testing as npt

from dipy.utils.logging import (
    CustomHandler,
    add_file_handler,
    attached_log_files,
    file_handler,
    get_logger,
    remove_file_handler,
)


def test_logger_stdout(caplog):
    logger = get_logger(name="dipy_test_stdout")
    with caplog.at_level(logging.INFO):
        logger.info("A message for stdout")
        logger.info("")
        logger.warning("A warning for stdout")
        logger.error("An error for stdout")
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
    log_out = caplog.text
    assert "A message for stdout" in log_out
    assert "\n" in log_out  # The empty line
    assert "A warning for stdout" in log_out
    assert "An error for stdout" in log_out


def test_logger_file(tmp_path):
    filename = tmp_path / "log.txt"
    logger = get_logger(name="dipy_test_file", filename=filename, force=True)
    logger.info("A message for file")
    logger.info("")
    logger.warning("A warning for file")
    logger.error("An error for file")
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    assert filename.exists()
    with open(filename, encoding="utf-8") as f:
        content = f.read()
    assert "A message for file" in content
    assert "\n" in content  # The empty line
    assert "A warning for file" in content
    assert "An error for file" in content


def test_custom_handler_writes_empty_line():
    stream = io.StringIO()
    handler = CustomHandler(stream=stream)
    logger = logging.getLogger("dipy_custom_handler_test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("First message")
    logger.info("")
    logger.info("Second message")
    handler.flush()
    output = stream.getvalue()
    assert "First message" in output
    assert "\n\n" in output  # One empty line between messages
    assert "Second message" in output


def test_file_handlers_are_listed_and_released(tmp_path):
    dipy_logger = logging.getLogger("dipy")
    before = list(dipy_logger.handlers)
    first, second = tmp_path / "first.log", tmp_path / "second.log"

    handler = add_file_handler(first, level=logging.WARNING)
    with file_handler(second, level=logging.DEBUG):
        npt.assert_equal(
            attached_log_files()[-2:],
            [(str(first), logging.WARNING), (str(second), logging.DEBUG)],
        )
    npt.assert_equal(attached_log_files()[-1], (str(first), logging.WARNING))

    remove_file_handler(handler=handler)
    npt.assert_equal(dipy_logger.handlers, before)
    npt.assert_equal(handler.stream.closed, True)
