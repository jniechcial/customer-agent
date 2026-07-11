"""Agent definition (OpenAI Agents SDK). One agent, one retrieval tool; the model
decides when and how often to search before answering."""

from agents import Agent, ModelSettings, set_default_openai_key
from openai.types.shared import Reasoning

from customer_agent.agent.prompts import SYSTEM_PROMPT
from customer_agent.agent.tools import search_knowledge_base
from customer_agent.config import get_settings


def build_agent(reasoning_summary: bool = False) -> Agent:
    settings = get_settings()
    # The SDK reads OPENAI_API_KEY from the environment; we source it from .env
    # via Settings instead, so hand it over explicitly.
    set_default_openai_key(settings.openai_api_key)
    # Reasoning models only emit summaries when asked; chat.py --show-steps wants them.
    model_settings = ModelSettings(reasoning=Reasoning(summary="auto")) if reasoning_summary else ModelSettings()
    return Agent(
        name="Wix Support Agent",
        instructions=SYSTEM_PROMPT,
        model=settings.agent_model,
        model_settings=model_settings,
        tools=[search_knowledge_base],
    )
