import pytest

from customer_agent.pricing import MODEL_PRICES, answer_cost_usd


def test_cost_applies_cached_discount():
    # 800 uncached @ $5/1M + 200 cached @ $0.50/1M + 100 out @ $30/1M
    assert answer_cost_usd("gpt-5.5", 1000, 200, 100) == pytest.approx(
        (800 * 5.00 + 200 * 0.50 + 100 * 30.00) / 1_000_000
    )


def test_unknown_model_costs_none():
    assert answer_cost_usd("some-future-model", 1000, 0, 100) is None


def test_agent_model_is_priced():
    """The configured agent model must have a price entry, or eval cost silently
    degrades to None."""
    from customer_agent.config import get_settings

    assert get_settings().agent_model in MODEL_PRICES
