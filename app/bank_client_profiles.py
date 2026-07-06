import json
from pathlib import Path

from pydantic import BaseModel, Field


class BankClientProfile(BaseModel):
    client_id: str
    profile_created_at: str
    name: str
    age: int
    income_type: str
    monthly_income_myr: float
    risk_tolerance: str
    dependents: int
    employment: str
    existing_products: list[dict] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    notes: str = ""
    history_seed: list[str] = Field(default_factory=list)


BANK_CLIENT_PROFILES_DIR = Path(__file__).resolve().parents[1] / "bank_client_profiles"


def list_bank_client_ids() -> list[str]:
    return sorted(path.stem for path in BANK_CLIENT_PROFILES_DIR.glob("*.json"))


def load_bank_client_profile(client_id: str) -> BankClientProfile:
    path = BANK_CLIENT_PROFILES_DIR / f"{client_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Bank client profile not found: {client_id}")

    profile_data = json.loads(path.read_text())
    return BankClientProfile.model_validate(profile_data)


def format_profile_for_prompt(profile: BankClientProfile) -> str:
    """
    Build working-memory text from profile.
    We intentionally exclude history_seed in Phase 0.
    """
    return (
        f"Client: {profile.name} ({profile.client_id})\n"
        f"Profile as of: {profile.profile_created_at}\n"
        f"Age: {profile.age} | Income: {profile.income_type}, "
        f"RM{profile.monthly_income_myr:,.0f}/month\n"
        f"Risk tolerance: {profile.risk_tolerance} | Dependents: {profile.dependents}\n"
        f"Employment: {profile.employment}\n"
        f"Products: {json.dumps(profile.existing_products, indent=2)}\n"
        f"Goals: {', '.join(profile.goals)}\n"
        f"Notes: {profile.notes}"
    )
