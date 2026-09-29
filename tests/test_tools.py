from app.tools import (
    calculate_expression,
    lookup_cve,
    reset_tool_log,
    get_tool_log,
)


def test_addition():
    reset_tool_log()

    result = calculate_expression("10 + 20")

    assert result["result"] == 30


def test_complex_calculation():
    reset_tool_log()

    result = calculate_expression("(12 + 8) * 5")

    assert result["result"] == 100


def test_power():
    reset_tool_log()

    result = calculate_expression("2 ** 8")

    assert result["result"] == 256


def test_invalid_expression():
    reset_tool_log()

    result = calculate_expression(
        "__import__('os').system('echo test')"
    )

    assert "error" in result


def test_invalid_cve_format():
    reset_tool_log()

    result = lookup_cve("INVALID-CVE")

    assert "error" in result


def test_tool_log():
    reset_tool_log()

    calculate_expression("5 * 5")

    logs = get_tool_log()

    assert len(logs) == 1
    assert logs[0]["tool"] == "calculate_expression"
    assert logs[0]["result"] == 25