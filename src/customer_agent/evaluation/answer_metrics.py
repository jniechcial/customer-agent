"""Answer metrics via deepeval GEval, judged by Claude (cross-vendor to the
agent to avoid self-preference bias).

Design decisions, from eyeballing judge outputs on ~10 baseline examples:

- Correctness is binary: an answer is either fully correct or it is not. Partial
  or incomplete answers count as incorrect — no partial credit inside the score.
- Partial answers get their own label instead: AnswerPartiallyCorrect flags
  answers that are not fully correct but not completely wrong, so they can be
  filtered and eyeballed separately (in Phoenix or the scores file) without
  polluting the correctness rate.
- Grounding is checked against sources, not against the reference's silence.
  The judge receives (as test-case context) the full text of the gold articles
  the agent actually retrieved — gold ∩ retrieved, built in scoring.py. Extras
  beyond the expected answer are fine when supported by those articles; specific
  claims found in neither the expected answer nor the articles are hallucinations
  and make the answer incorrect. Gold articles the agent never retrieved are
  deliberately withheld: matching them is parametric memory, not grounding.
- Judge reasoning explains failures only: correct answers are accepted silently,
  with no callouts of what is right.
- Together the two metrics read: correctness=1 → correct; correctness=0 with
  partial=1 → on the right track; 0/0 → wrong (contradiction, different problem,
  or predominantly ungrounded).

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
answer. The expected answer is the ground truth written by a support expert. The
context contains the knowledge-base articles the expected answer is based on (when
available) — the only legitimate sources for claims that go beyond the expected
answer.

The verdict is binary: fully correct, or not. A partially correct or incomplete
answer is NOT correct.

The actual output is correct only if ALL of the following hold:
- It contains every key fact and every essential step of the expected answer.
  Wording, ordering, formatting, and level of detail are free to differ;
  conversational tone vs. procedural markdown does not matter.
- It does not contradict the expected answer or the context articles. If the
  expected answer says something is impossible or unsupported, the actual output
  must not claim it is possible.
- Every specific factual claim that goes beyond the expected answer — a numeric
  limit, price, plan or feature availability, setting or feature name, extra step
  or method — is supported by the context articles. Extras supported by the
  context articles are NOT a fault, however much they add. A specific claim found
  in neither the expected answer nor the context articles makes the answer
  incorrect.

Reasoning: never call out what the answer gets right. If the answer is correct,
say only "correct". If it is incorrect, state briefly and only what is wrong:
the contradiction, the missing essential step(s), or the ungrounded fact(s).
"""

PARTIAL_CRITERIA = """\
Flag whether the actual output is ON THE RIGHT TRACK relative to the expected
answer (the ground truth written by a support expert) without being fully
correct. The context contains the knowledge-base articles the expected answer
is based on (when available).

Score 1 when the answer addresses the user's actual problem and does not
contradict the expected answer or the context articles, but falls short of
fully correct in either or both of these ways:
- it covers some but not all key facts or essential steps of the expected
  answer, or
- it covers all essentials but states specific factual claims (numeric limits,
  prices, feature availability, setting or feature names) found in neither the
  expected answer nor the context articles.

Score 0 in both other cases:
- fully correct: every essential is covered and every extra specific claim is
  supported by the expected answer or the context articles; or
- wrong: it contradicts the expected answer, addresses a different problem or
  procedure, or its content is predominantly unsupported.

Reasoning: one short sentence — what is missing or unsupported (if on the right
track), or why the answer is not flagged (fully correct / wrong).
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
    LLMTestCaseParams.CONTEXT,
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
