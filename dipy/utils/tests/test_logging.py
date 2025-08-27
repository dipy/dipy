import io
import logging

from dipy.utils.logging import CustomHandler, get_logger


def test_logger_stdout(caplog):
    logger = get_logger("dipy_test_stdout")
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
    logger = get_logger("dipy_test_file", filename=filename, force=True)
    logger.info("A message for file")
    logger.info("")
    logger.warning("A warning for file")
    logger.error("An error for file")
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    assert filename.exists()
    with open(filename, "r") as f:
        content = f.read()
    assert "A message for file" in content
    assert "\n" in content  # The empty line
    assert "A warning for file" in content
    assert "An error for file" in content


def test_custom_handler_writes_empty_line(monkeypatch):
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
