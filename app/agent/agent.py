from ollama import Client

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.bank_client_profiles import BankClientProfile, format_profile_for_prompt
from app.config import settings


def build_messages(
    profile: BankClientProfile,
    session_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Assemble working memory: system rules + profile + this session only."""
    system = (
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        f"--- BANK CLIENT PROFILE ---\n"
        f"{format_profile_for_prompt(profile)}"
    )
    return [{"role": "system", "content": system}, *session_messages]


def generate_reply(
    profile: BankClientProfile,
    session_messages: list[dict[str, str]],
) -> str:
    client = Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.ollama_model,
        messages=build_messages(profile, session_messages),
    )
    return response["message"]["content"]
