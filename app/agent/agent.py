from ollama import Client

from app.agent.prompts import AGENT_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT_WITH_MEMORY
from app.bank_client_profiles import BankClientProfile, format_profile_for_prompt
from app.config import settings


def build_messages(
    profile: BankClientProfile,
    session_messages: list[dict[str, str]],
    episodic_context: str | None = None,
) -> list[dict[str, str]]:
    """Assemble working memory: system rules + profile + retrieved episodes + this session only."""
    base_prompt = AGENT_SYSTEM_PROMPT_WITH_MEMORY if episodic_context else AGENT_SYSTEM_PROMPT
    system = (
        f"{base_prompt}\n\n"
        f"--- BANK CLIENT PROFILE ---\n"
        f"{format_profile_for_prompt(profile)}"
    )
    if episodic_context:
        system += f"\n\n--- RELEVANT PAST CONTEXT (recalled from memory) ---\n{episodic_context}"
    return [{"role": "system", "content": system}, *session_messages]


def generate_reply(
    profile: BankClientProfile,
    session_messages: list[dict[str, str]],
    episodic_context: str | None = None,
) -> str:
    client = Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.ollama_model,
        messages=build_messages(profile, session_messages, episodic_context),
    )
    return response["message"]["content"]
