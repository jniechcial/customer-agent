"""Answer correctness via deepeval GEval, judged by Claude (cross-vendor to the
agent to avoid self-preference bias).

Known validity risk: dataset answers are step-by-step markdown procedures while
the agent answers conversationally — the criteria below deliberately target
factual/procedural agreement, not style. Iterate here (and hand-check ~10
examples) before trusting absolute numbers.
"""

import sys

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from customer_agent.config import get_settings

CORRECTNESS_CRITERIA = """\
Judge whether the actual output gives the user the same resolution as the expected
answer. The expected answer is the ground truth written by a support expert.

- The actual output is correct if it contains the key facts and the essential steps
  of the expected answer, even if wording, ordering, formatting, or level of detail
  differ.
- Ignore style: conversational tone vs. procedural markdown does not matter.
- Penalize contradictions of the expected answer, missing steps that are essential
  to resolving the issue, and fabricated features/settings/steps.
- Extra correct-and-relevant information beyond the expected answer is not a fault.
- If the expected answer says something is impossible or unsupported, the actual
  output must not claim it is possible.
"""


def _judge_model():
    settings = get_settings()
    if settings.anthropic_api_key:
        from deepeval.models import AnthropicModel

        # No temperature: Claude 5 models reject the deprecated param, and deepeval
        # only sends it when explicitly configured.
        # Thinking disabled: Claude 5 thinks adaptively by default, and deepeval's
        # response parsing reads content[0].text, which crashes when a thinking
        # block precedes the text block.
        return AnthropicModel(
            model=settings.judge_model,
            generation_kwargs={"thinking": {"type": "disabled"}},
        )
    print(
        "WARNING: ANTHROPIC_API_KEY not set - falling back to deepeval's default "
        "OpenAI judge. Add the key to .env to use the configured Claude judge.",
        file=sys.stderr,
    )
    return None  # deepeval default (OpenAI)


def make_correctness_metric() -> GEval:
    return GEval(
        name="AnswerCorrectness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=_judge_model(),
        threshold=0.5,
    )
