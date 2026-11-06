"""Router behavior with a fake model. These tests make no network calls and need no API key."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from main import MAX_INPUT_CHARS, MessageClassifier, RouterError, chat_once, run_once, validate_input


class FakeLLM:
    def __init__(self, route="logical", fail_on=None):
        self.route = route
        self.fail_on = fail_on
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is MessageClassifier
        return SimpleNamespace(invoke=self._classify)

    def _classify(self, messages):
        self.calls.append(("classify", messages))
        if self.fail_on == "classify":
            raise ConnectionError("provider down")
        return MessageClassifier(message_type=self.route)

    def invoke(self, messages):
        self.calls.append(("respond", messages))
        if self.fail_on == "respond":
            raise TimeoutError("provider timeout")
        return SimpleNamespace(content=f"reply to: {messages[-1]['content']}")


def test_emotional_message_goes_to_supportive_node():
    llm = FakeLLM(route="emotional")
    result = run_once("I feel overwhelmed", llm)
    assert result["route"] == "supportive"
    assert "not a therapist" in llm.calls[1][1][0]["content"]


def test_logical_message_goes_to_logical_node():
    llm = FakeLLM(route="logical")
    result = run_once("What is the capital of France?", llm)
    assert result["route"] == "logical"
    assert "logical assistant" in llm.calls[1][1][0]["content"]


def test_exactly_two_model_calls_per_message():
    llm = FakeLLM()
    run_once("hello", llm)
    assert [name for name, _ in llm.calls] == ["classify", "respond"]


def test_reply_text_is_returned():
    assert chat_once("hello", FakeLLM()) == "reply to: hello"


def test_calls_are_single_turn():
    """The response node sees only the current message: this prototype has no memory."""
    llm = FakeLLM()
    run_once("first", llm)
    run_once("second", llm)
    last_respond = [messages for name, messages in llm.calls if name == "respond"][-1]
    assert [m["role"] for m in last_respond] == ["system", "user"]
    assert "first" not in str(last_respond)


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_is_rejected_before_any_model_call(text):
    llm = FakeLLM()
    with pytest.raises(ValueError, match="empty"):
        run_once(text, llm)
    assert not llm.calls


def test_too_long_input_is_rejected():
    with pytest.raises(ValueError, match="longer"):
        validate_input("x" * (MAX_INPUT_CHARS + 1))


def test_input_at_limit_is_accepted():
    assert validate_input("x" * MAX_INPUT_CHARS)


def test_input_is_stripped():
    assert validate_input("  hi  ") == "hi"


@pytest.mark.parametrize("stage", ["classify", "respond"])
def test_provider_failure_becomes_router_error(stage):
    with pytest.raises(RouterError, match="provider failed"):
        run_once("hello", FakeLLM(fail_on=stage))


def test_invalid_classification_is_rejected_by_the_schema():
    with pytest.raises(ValueError):
        MessageClassifier(message_type="other")


def test_arabic_input_passes_through():
    assert "مرحبا" in chat_once("مرحبا", FakeLLM())


def test_cli_reports_invalid_input(capsys):
    assert main.main(["   "]) == 2
    assert "Invalid input" in capsys.readouterr().err


def test_cli_reports_provider_failure(monkeypatch, capsys):
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM(fail_on="classify"))
    assert main.main(["hello"]) == 1
    assert "provider failed" in capsys.readouterr().err


def test_cli_prints_route_and_reply(monkeypatch, capsys):
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM())
    assert main.main(["hello"]) == 0
    assert "[route: logical]" in capsys.readouterr().out


def test_import_needs_no_api_key():
    code = "import main; print(main.DEFAULT_MODEL)"
    env = {k: v for k, v in __import__("os").environ.items() if "ANTHROPIC" not in k}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
