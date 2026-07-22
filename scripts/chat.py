"""Open-ended multi-turn terminal chat with the support agent.

Usage:
    uv run python scripts/chat.py [--show-steps]

Type your question; 'exit', 'quit', or Ctrl-D to leave.
--show-steps renders the agent's interim steps (tool calls with their arguments,
reasoning summaries if the model emits them) in gray as they happen.
"""

import argparse
import asyncio
import warnings

from agents import Agent, Runner
from agents.result import RunResultStreaming
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text

from customer_agent.tracing import setup_tracing


async def run_turn_streamed(agent: Agent, input_items: list, console: Console) -> RunResultStreaming:
    """Run one turn, rendering interim steps in gray as they stream in."""
    result = Runner.run_streamed(agent, input_items)
    async for event in result.stream_events():
        if event.type != "run_item_stream_event":
            continue
        item = event.item
        if item.type == "reasoning_item":
            for part in item.raw_item.summary:
                console.print(Text("·", style="dim"))
                console.print(Padding(Text(part.text, style="dim"), (0, 0, 0, 7)))
        elif item.type == "tool_call_item":
            call = item.raw_item
            name = getattr(call, "name", type(call).__name__)
            arguments = getattr(call, "arguments", "")
            console.print(Text(f"⚙ {name}", style="dim"))
            console.print(Padding(Text(arguments, style="dim"), (0, 0, 0, 7)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="render interim steps (tool calls, reasoning) in gray as the agent works",
    )
    args = parser.parse_args()

    setup_tracing("chat")
    from customer_agent.agent.agent import build_agent
    from customer_agent.agent.tools import search_budget
    from customer_agent.config import get_settings

    # openinference-instrumentation-openai (<=0.1.52) calls response.parse() synchronously
    # on streamed async responses, leaving an unawaited coroutine behind. Harmless for us
    # (only the raw-LLM span loses response attributes; agent spans are unaffected).
    # Registered after the imports above: a transitive dep prepends a catch-all filter
    # during import, which would otherwise shadow this one.
    warnings.filterwarnings("ignore", message="coroutine 'AsyncAPIResponse.parse' was never awaited")

    console = Console()
    agent = build_agent(reasoning_summary=args.show_steps)
    settings = get_settings()
    console.print(
        f"[bold]Wix support agent[/bold] (model={settings.agent_model}, "
        f"collection={settings.collection_name}). Ctrl-D or 'exit' to quit.\n"
    )

    conversation: list = []
    try:
        while True:
            try:
                user_input = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            turn_input = conversation + [{"role": "user", "content": user_input}]
            # Fresh search budget each user turn; asyncio.run and run_sync both
            # copy the current context, so the budget reaches the tool calls.
            with search_budget():
                if args.show_steps:
                    result = asyncio.run(run_turn_streamed(agent, turn_input, console))
                else:
                    result = Runner.run_sync(agent, turn_input)
            conversation = result.to_input_list()
            console.print()
            console.print("[bold green]agent>[/bold green]")
            console.print(Padding(Markdown(str(result.final_output)), (0, 0, 0, 7)))
            console.print()
    finally:
        from customer_agent.retrieval.pipeline import close_pipeline

        close_pipeline()

    console.print("bye")


if __name__ == "__main__":
    main()
