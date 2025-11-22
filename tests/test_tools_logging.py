import logging

import pytest

from src.graph.tools.decorators import create_logged_tool, log_io


class DummyTool:
    def __init__(self):
        self.runs = []

    def _run(self, x):
        self.runs.append(x)
        return x + 1


def test_log_io_decorator_logs_inputs_and_outputs(caplog):
    @log_io
    def add(a, b):
        return a + b

    with caplog.at_level(logging.INFO):
        result = add(1, 2)

    assert result == 3
    assert any("Tool add called with parameters" in record.message for record in caplog.records)
    assert any("Tool add returned: 3" in record.message for record in caplog.records)


def test_create_logged_tool_wraps_run_and_logs(caplog):
    LoggedDummy = create_logged_tool(DummyTool)
    tool = LoggedDummy()

    with caplog.at_level(logging.INFO):
        output = tool._run(5)

    assert output == 6
    # Original behavior should still be recorded on the instance
    assert tool.runs == [5]
    # Logging statements emitted from mixin
    assert any("DummyTool._run called with parameters: 5" in record.message for record in caplog.records)
    assert any("DummyTool returned: 6" in record.message for record in caplog.records)
