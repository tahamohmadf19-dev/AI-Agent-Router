"""Single-turn message router built with LangGraph.

Flow: classify the message, then send it to one of two prompt-specialized response nodes.
Each call starts with a fresh state, so there is no memory between calls.
"""
import argparse
import logging
import os
import sys
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger("agent_router")

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5"
MAX_INPUT_CHARS = 4000

CLASSIFIER_PROMPT = """Classify the user message as either:
- 'emotional': it seeks emotional support or deals with feelings or personal struggles
- 'logical': it asks for facts, information, analysis, or practical solutions
If a message contains both, choose the one the person most needs answered first."""

SUPPORTIVE_PROMPT = """You are a supportive listener. Acknowledge the person's feelings and ask a
thoughtful question that helps them reflect. You are not a therapist or a medical professional:
do not diagnose or give treatment advice, and suggest a qualified professional for ongoing
difficulties. If the person may be in danger or mentions self-harm, encourage them to contact
local emergency services or a crisis line right away."""

LOGICAL_PROMPT = """You are a logical assistant. Give clear, concise answers based on facts and
reasoning. If you are not sure, say so rather than guessing."""


class RouterError(RuntimeError):
    """The model provider failed. The message is safe to show to a user."""


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ..., description="Whether the message needs a supportive or a logical response."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: Optional[str]
    next: Optional[str]


def get_llm() -> Any:
    """Create the chat model only when needed, so importing this module needs no API key."""
    from langchain.chat_models import init_chat_model

    load_dotenv()
    return init_chat_model(os.getenv("ROUTER_MODEL", DEFAULT_MODEL))


def validate_input(user_input: str) -> str:
    text = (user_input or "").strip()
    if not text:
        raise ValueError("Message is empty.")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"Message is longer than {MAX_INPUT_CHARS} characters.")
    return text


def build_graph(llm: Any) -> Any:
    """Build the routing graph around a chat model, which tests can replace with a fake."""

    def classify_message(state: State) -> dict:
        last_message = state["messages"][-1]
        classifier = llm.with_structured_output(MessageClassifier)
        result = classifier.invoke(
            [
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": last_message.content},
            ]
        )
        return {"message_type": result.message_type}

    def route(state: State) -> dict:
        return {"next": "supportive" if state.get("message_type") == "emotional" else "logical"}

    def respond(system_prompt: str, state: State) -> dict:
        last_message = state["messages"][-1]
        reply = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": last_message.content},
            ]
        )
        return {"messages": [{"role": "assistant", "content": reply.content}]}

    builder = StateGraph(State)
    builder.add_node("classifier", classify_message)
    builder.add_node("router", route)
    builder.add_node("supportive", lambda state: respond(SUPPORTIVE_PROMPT, state))
    builder.add_node("logical", lambda state: respond(LOGICAL_PROMPT, state))
    builder.add_edge(START, "classifier")
    builder.add_edge("classifier", "router")
    builder.add_conditional_edges(
        "router", lambda state: state.get("next"), {"supportive": "supportive", "logical": "logical"}
    )
    builder.add_edge("supportive", END)
    builder.add_edge("logical", END)
    return builder.compile()


def run_once(user_input: str, llm: Optional[Any] = None) -> dict:
    """Classify and answer one message. Returns the route and the reply."""
    text = validate_input(user_input)
    graph = build_graph(llm if llm is not None else get_llm())
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": text}], "message_type": None, "next": None}
        )
    except Exception as exc:  # Provider SDKs raise many types; callers need just one.
        logger.error("Model call failed: %s", type(exc).__name__)
        raise RouterError(f"The model provider failed ({type(exc).__name__}).") from exc
    return {"route": result["next"], "reply": result["messages"][-1].content}


def chat_once(user_input: str, llm: Optional[Any] = None) -> str:
    return run_once(user_input, llm)["reply"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Route one message to a response node.")
    parser.add_argument("message", help="The user message")
    args = parser.parse_args(argv)
    try:
        result = run_once(args.message)
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2
    except RouterError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"[route: {result['route']}]\n{result['reply']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
