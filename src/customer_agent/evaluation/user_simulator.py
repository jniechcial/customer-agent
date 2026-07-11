"""User-side of an eval conversation. v1 is single-turn: ask the dataset question, stop.

LATER: an LLM-backed simulator that plays the customer — answers the agent's
clarifying questions grounded in the dataset row, gives follow-ups, and ends the
conversation when satisfied. The runner already drives conversations through this
interface, so that lands as a new class + max_turns > 1, no runner rewrite.
"""

from typing import Protocol


class UserSimulator(Protocol):
    def first_message(self, row: dict) -> str:
        """Opening user message for a dataset row."""
        ...

    def next_message(self, row: dict, agent_reply: str, turn: int) -> str | None:
        """Follow-up message, or None to end the conversation."""
        ...


class ScriptedSingleTurn:
    def first_message(self, row: dict) -> str:
        return row["question"]

    def next_message(self, row: dict, agent_reply: str, turn: int) -> str | None:
        return None


def get_default_simulator() -> UserSimulator:
    return ScriptedSingleTurn()
