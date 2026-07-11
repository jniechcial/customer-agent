"""Model pricing (USD per 1M tokens) for answer-cost estimates on eval runs.

Manually maintained snapshot of https://platform.openai.com/docs/pricing
(checked 2026-07-11). Unknown models yield None rather than a wrong number.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5.5": ModelPrice(input_per_1m=5.00, cached_input_per_1m=0.50, output_per_1m=30.00),
}


def answer_cost_usd(
    model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int
) -> float | None:
    """Cost of one answer. Cached input tokens are a subset of input_tokens and
    billed at the discounted rate."""
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    uncached_input = input_tokens - cached_input_tokens
    return (
        uncached_input * price.input_per_1m
        + cached_input_tokens * price.cached_input_per_1m
        + output_tokens * price.output_per_1m
    ) / 1_000_000
