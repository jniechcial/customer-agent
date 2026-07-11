"""Run the eval: generate agent answers over a split, then score with deepeval.

Usage:
    uv run python scripts/run_eval.py --split validation [--subset wixqa_expertwritten]
                                      [--limit N] [--rescore runs/<id>.jsonl]

Two phases:
  1. generate — run the agent per question, persist runs/<run_id>.jsonl (skipped with --rescore)
  2. score    — deepeval: GEval answer correctness + deterministic retrieval metrics
Summary is printed and written to runs/<run_id>.summary.json.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from customer_agent.tracing import setup_tracing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--subset", default=None, help="QA subset (default from config)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--rescore", type=Path, default=None,
                        help="skip generation; score an existing runs/<id>.jsonl")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.rescore:
        artifact = args.rescore
    else:
        import asyncio

        from customer_agent.evaluation.runner import generate_run

        run_name = args.run_id or f"{args.split}-{os.getpid()}"
        setup_tracing(f"eval-{run_name}")
        artifact = asyncio.run(
            generate_run(args.split, subset=args.subset, limit=args.limit, run_id=args.run_id)
        )
        print(f"generation done -> {artifact}")

    from customer_agent.evaluation.scoring import score

    summary = score(artifact)
    summary_path = artifact.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {artifact.name}: n={summary['n']}, "
          f"avg tool calls={summary['avg_tool_calls']:.2f} ===")
    if summary["usage"]:
        u = summary["usage"]
        print(f"  tokens: in={u['total_input_tokens']:,} "
              f"(cached {u['total_cached_input_tokens']:,}) "
              f"out={u['total_output_tokens']:,} avg/question={u['avg_total_tokens']:,.0f}")
    if summary["cost_usd"]:
        c = summary["cost_usd"]
        print(f"  cost: total=${c['total']:.4f} avg/answer=${c['avg_per_answer']:.4f}")
    if summary["latency_seconds"]:
        lat = summary["latency_seconds"]
        print(f"  latency: avg={lat['avg']:.1f}s max={lat['max']:.1f}s")
    for name, stats in summary["metrics"].items():
        print(f"  {name:<24} {stats['mean']:.3f}")
    print(f"summary -> {summary_path}")
    print(f"per-question scores (judge reasoning) -> {summary['scores_file']}")
    if summary["phoenix_annotations"]:
        from customer_agent.config import get_settings

        phoenix_ui = get_settings().phoenix_endpoint.split("/v1/")[0]
        print(f"{summary['phoenix_annotations']} annotations -> {phoenix_ui} "
              "(open the run's eval project; sort/filter traces by AnswerCorrectness, "
              "judge reasoning is in each trace's annotations pane)")


if __name__ == "__main__":
    main()
