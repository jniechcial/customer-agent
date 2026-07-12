"""Answer metrics via deepeval GEval, judged by Claude (cross-vendor to the
agent to avoid self-preference bias).

Design decisions, from eyeballing judge outputs on ~10 baseline examples:

- Correctness is binary: an answer is either fully correct or it is not. Partial
  or incomplete answers count as incorrect — no partial credit inside the score.
- Partial answers get their own label instead: AnswerPartiallyCorrect flags
  answers that are not fully correct but not completely wrong, so they can be
  filtered and eyeballed separately (in Phoenix or the scores file) without
  polluting the correctness rate.
- Grounding is asymmetric. Stylistic freedom is fine — different ordering, level
  of detail, creative phrasing, extra generic guidance. But concrete factual
  claims not grounded in the expected answer (limits, prices, feature
  availability) make the answer incorrect even when everything else matches.
- Judge reasoning explains failures only: correct answers are accepted silently,
  with no callouts of what is right.

Known validity risk: dataset answers are step-by-step markdown procedures while
the agent answers conversationally — the criteria below deliberately target
factual/procedural agreement, not style. Hand-check ~10 examples after every
criteria change before trusting absolute numbers.
"""

import sys

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from customer_agent.config import get_settings

CORRECTNESS_CRITERIA = """\
Judge whether the actual output gives the user the full resolution of the expected
answer. The expected answer is the ground truth written by a support expert.
The verdict is binary: fully correct, or not. A partially correct or incomplete
answer is NOT correct.

The actual output is correct only if ALL of the following hold:
- It contains every key fact and every essential step of the expected answer.
  Wording, ordering, formatting, and level of detail are free to differ;
  conversational tone vs. procedural markdown does not matter.
- It does not contradict the expected answer. If the expected answer says
  something is impossible or unsupported, the actual output must not claim it
  is possible.
- It states no ungrounded specific facts: any concrete factual claim that is not
  supported by the expected answer — numeric limits, prices, plan or feature
  availability, names of settings or features — makes the answer incorrect,
  even if the rest matches. Extra guidance that is merely more creative, more
  detailed, or differently ordered than the expected answer is not a fault;
  only unsupported specific factual claims are.

Reasoning: never call out what the answer gets right. If the answer is correct,
say only "correct". If it is incorrect, state briefly and only what is wrong:
the contradiction, the missing essential step(s), or the ungrounded fact(s).
"""

PARTIAL_CRITERIA = """\
Flag whether the actual output is a PARTIALLY correct answer relative to the
expected answer (the ground truth written by a support expert).

Score 1 only when the answer is partially correct or incomplete: it covers some
key facts or essential steps of the expected answer, or points the user in the
right direction, but omits other essentials or stops short of the full
resolution — while NOT contradicting the expected answer and NOT stating
specific facts (numeric limits, prices, feature availability) that the expected
answer does not support.

Score 0 in both other cases: the answer is fully correct (nothing essential
missing), or the answer is wrong (contradicts the expected answer, states
ungrounded specific facts, or addresses a different problem).

Reasoning: one short sentence — what essential part is missing (if partial), or
why the answer is not partial (fully correct / wrong).
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


_EVALUATION_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]


def make_correctness_metric(model=None) -> GEval:
    # strict_mode makes the score binary: 1 for fully correct, 0 for anything less.
    return GEval(
        name="AnswerCorrectness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=_EVALUATION_PARAMS,
        model=model if model is not None else _judge_model(),
        strict_mode=True,
    )


def make_partial_metric(model=None) -> GEval:
    # Binary flag, not a quality score: 1 = partially correct/incomplete,
    # 0 = fully correct OR wrong. Run mean = fraction of partial answers.
    return GEval(
        name="AnswerPartiallyCorrect",
        criteria=PARTIAL_CRITERIA,
        evaluation_params=_EVALUATION_PARAMS,
        model=model if model is not None else _judge_model(),
        strict_mode=True,
    )


def answer_metrics() -> list[GEval]:
    model = _judge_model()
    return [make_correctness_metric(model), make_partial_metric(model)]
