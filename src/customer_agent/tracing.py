"""Phoenix tracing bootstrap. Call setup_tracing() once per entry point, before any agent/OpenAI use.

Sink is Phoenix only: the Agents SDK's default export to the OpenAI platform is cleared by
replacing its trace processors before the OpenInference instrumentor registers its own.
"""

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from phoenix.otel import register

from customer_agent.config import get_settings

_initialized = False


def setup_tracing(project_name: str) -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    settings = get_settings()

    # Drop the Agents SDK default processor (exports to the OpenAI platform) before
    # instrumenting, so Phoenix is the only sink. Order matters: set_trace_processors
    # after instrumenting would wipe the OpenInference processor instead.
    from agents import set_trace_processors

    set_trace_processors([])

    tracer_provider = register(
        project_name=project_name,
        endpoint=settings.phoenix_endpoint,
        batch=True,
        set_global_tracer_provider=True,
        verbose=False,
    )
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
