"""
Test 05 — Episodic top-3 recall eval

Goal:
1) Plant many client_a (and a few client_b) turns in Chroma
2) Run a fixed probe set of questions
3) Score top-3 recall = correct episode keywords appear in any of top 3 hits
4) Score isolation = client_b search never returns client_a text

This measures retrieval only (Chroma), not the chat agent.

Run (from project root):
    uv run python -m tests.test_05_episodic_top3_recall

Uses: ./chroma_data_test
"""

from __future__ import annotations

from app.memory.episodic import clear_episodes, search_episodes, store_episode
from tests.helpers import print_header, print_subheader

TEST_CHROMA_PATH = "./chroma_data_test"
TOP_K = 3


# Planted Session-style turns for client_a
PLANTED_A: list[dict[str, str]] = [
    {
        "user": "I'm thinking about refinancing my home loan. What about break costs?",
        "assistant": "Compare break costs against monthly savings before you refinance.",
        "tag": "break_costs",
    },
    {
        "user": "My current home loan rate is about 4.2 percent.",
        "assistant": "A 4.2 percent rate is useful context when comparing refinance offers.",
        "tag": "rate_4_2",
    },
    {
        "user": "I might get a bonus in Q4 and want to prepay part of the loan.",
        "assistant": "A Q4 bonus used for prepayment can reduce interest if there is no penalty.",
        "tag": "bonus_prepay",
    },
    {
        "user": "I also want to build an education fund for my children.",
        "assistant": "An education fund can sit alongside your home loan goals.",
        "tag": "education_fund",
    },
    {
        "user": "I have two dependents and I am a salaried senior engineer.",
        "assistant": "Two dependents and salaried income affect how aggressive your plan can be.",
        "tag": "dependents_salary",
    },
    {
        "user": "My outstanding home loan is around 420000 with Maybank.",
        "assistant": "A Maybank outstanding balance near 420000 is the base for refinance maths.",
        "tag": "maybank_420k",
    },
    {
        "user": "I am worried about early settlement fees on my loan.",
        "assistant": "Early settlement fees are another cost alongside break costs.",
        "tag": "settlement_fees",
    },
    {
        "user": "Could I switch to a shorter tenure if the rate drops?",
        "assistant": "A shorter tenure raises the monthly payment but cuts total interest.",
        "tag": "shorter_tenure",
    },
    {
        "user": "I have unit trusts worth about 85000 in a mixed asset fund.",
        "assistant": "Your 85000 unit trust holding is separate from the home loan decision.",
        "tag": "unit_trusts",
    },
    {
        "user": "My risk tolerance is moderate, not aggressive.",
        "assistant": "Moderate risk tolerance argues against highly leveraged refinance moves.",
        "tag": "moderate_risk",
    },
    {
        "user": "Would offset account features help with interest?",
        "assistant": "An offset account can reduce interest by parking savings against the loan.",
        "tag": "offset_account",
    },
    {
        "user": "I prefer fixed rate for peace of mind for the next few years.",
        "assistant": "A fixed rate trades flexibility for payment certainty.",
        "tag": "fixed_rate",
    },
    {
        "user": "Can I refinance to free cash for renovation?",
        "assistant": "Cash-out refinance for renovation raises your outstanding balance.",
        "tag": "renovation_cashout",
    },
    {
        "user": "How does EPF withdrawal interact with my housing plans?",
        "assistant": "EPF housing rules are separate from bank refinance break costs.",
        "tag": "epf_housing",
    },
    {
        "user": "I want lower monthly payments even if total interest is a bit higher.",
        "assistant": "Lower monthly payments usually means a longer tenure tradeoff.",
        "tag": "lower_monthly",
    },
    {
        "user": "Is now a good time given the overnight policy rate?",
        "assistant": "OPR moves affect floating home loan packages more than fixed ones.",
        "tag": "opr_timing",
    },
    {
        "user": "I shared that my partner also contributes to household income.",
        "assistant": "Dual income can support a slightly higher repayment safely.",
        "tag": "dual_income",
    },
    {
        "user": "Please remind me we discussed insurance on the outstanding loan.",
        "assistant": "Mortgage reducing term insurance can cover the outstanding balance.",
        "tag": "mortgage_insurance",
    },
    {
        "user": "I do not want to sell my unit trusts just to pay the loan early.",
        "assistant": "Keeping the unit trusts avoids forced selling while you refinance.",
        "tag": "keep_unit_trusts",
    },
    {
        "user": "My goal is to clear the home loan before I turn 55.",
        "assistant": "Clearing the loan before 55 sets a concrete long-term payoff target.",
        "tag": "clear_before_55",
    },
]

PLANTED_B: list[dict[str, str]] = [
    {
        "user": "I am self-employed and prefer conservative investments.",
        "assistant": "Conservative allocations suit self-employed income volatility.",
        "tag": "b_conservative",
    },
    {
        "user": "I want capital preservation more than high growth.",
        "assistant": "Capital preservation should lead your portfolio choices.",
        "tag": "b_preservation",
    },
]

# Each probe: question + which keyword(s) must appear in some top-3 hit text
PROBES_A: list[dict] = [
    {
        "question": "What did we discuss about break costs when refinancing?",
        "must_any": ["break cost"],
    },
    {
        "question": "What home loan interest rate did I mention?",
        "must_any": ["4.2"],
    },
    {
        "question": "Did I talk about using a bonus to prepay the loan?",
        "must_any": ["bonus", "prepay"],
    },
    {
        "question": "What education goal did I mention for my children?",
        "must_any": ["education"],
    },
    {
        "question": "How many dependents did I say I have?",
        "must_any": ["two dependents", "2 dependents", "dependents"],
    },
    {
        "question": "Which bank and outstanding balance did we discuss?",
        "must_any": ["maybank", "420000", "420,000"],
    },
    {
        "question": "What did we say about early settlement fees?",
        "must_any": ["settlement"],
    },
    {
        "question": "Did we talk about switching to a shorter loan tenure?",
        "must_any": ["shorter tenure", "tenure"],
    },
    {
        "question": "What unit trust amount did I mention?",
        "must_any": ["85000", "85,000", "unit trust"],
    },
    {
        "question": "What risk tolerance did I say I have?",
        "must_any": ["moderate"],
    },
    {
        "question": "What did we discuss about an offset account?",
        "must_any": ["offset"],
    },
    {
        "question": "Did I say I prefer a fixed rate?",
        "must_any": ["fixed rate", "fixed"],
    },
    {
        "question": "Can I refinance to get cash for renovation?",
        "must_any": ["renovation", "cash-out", "cash out"],
    },
    {
        "question": "What did we say about EPF and housing?",
        "must_any": ["epf"],
    },
    {
        "question": "Did I prefer lower monthly payments?",
        "must_any": ["lower monthly", "monthly payment"],
    },
    {
        "question": "How does the overnight policy rate relate to my loan?",
        "must_any": ["opr", "overnight policy"],
    },
    {
        "question": "Did I mention dual household income?",
        "must_any": ["dual income", "partner"],
    },
    {
        "question": "What insurance topic related to the loan did we cover?",
        "must_any": ["insurance", "mortgage"],
    },
    {
        "question": "Did I want to keep my unit trusts instead of selling them?",
        "must_any": ["unit trust", "selling"],
    },
    {
        "question": "When do I want the home loan cleared by?",
        "must_any": ["55"],
    },
]

# Same-style questions asked as client_b must not surface client_a refinance content
ISOLATION_PROBES: list[str] = [
    "What did we discuss about break costs when refinancing?",
    "What home loan interest rate did I mention?",
    "Did I talk about using a bonus to prepay the loan?",
]


def _hit_matches(text: str, must_any: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in must_any)


def _top3_hit(hits: list[dict], must_any: list[str]) -> bool:
    return any(_hit_matches(h["text"], must_any) for h in hits[:TOP_K])


def main() -> None:
    print_header("TEST 05 — EPISODIC TOP-3 RECALL EVAL")
    clear_episodes(path=TEST_CHROMA_PATH)
    print(f"Chroma path: {TEST_CHROMA_PATH}")
    print(f"Planted client_a turns: {len(PLANTED_A)}")
    print(f"Planted client_b turns: {len(PLANTED_B)}")
    print(f"Probes: {len(PROBES_A)} | top_k={TOP_K}")

    print_header("STEP 1 — Plant episodes")
    for item in PLANTED_A:
        store_episode(
            "client_a",
            "session_a_eval",
            item["user"],
            item["assistant"],
            path=TEST_CHROMA_PATH,
        )
    for item in PLANTED_B:
        store_episode(
            "client_b",
            "session_b_eval",
            item["user"],
            item["assistant"],
            path=TEST_CHROMA_PATH,
        )
    print("Planting done.")

    print_header("STEP 2 — Score top-3 recall (client_a)")
    correct = 0
    failures: list[str] = []

    for index, probe in enumerate(PROBES_A, start=1):
        hits = search_episodes(
            probe["question"],
            "client_a",
            n_results=TOP_K,
            path=TEST_CHROMA_PATH,
        )
        ok = _top3_hit(hits, probe["must_any"])
        mark = "PASS" if ok else "FAIL"
        if ok:
            correct += 1
        else:
            preview = hits[0]["text"][:80].replace("\n", " ") if hits else "(no hits)"
            failures.append(f"{index}. {probe['question'][:60]}... | top1={preview}")

        print(f"[{index:02d}] {mark}  {probe['question'][:70]}")

    total = len(PROBES_A)
    recall = (correct / total) * 100 if total else 0.0

    print_subheader("TOP-3 RECALL RESULT")
    print(f"correct: {correct}/{total}")
    print(f"top-3 recall: {recall:.1f}%")

    if failures:
        print_subheader("FAILURES")
        for line in failures:
            print(f"- {line}")

    print_header("STEP 3 — Isolation (client_b must not get client_a)")
    leaks = 0
    a_markers = ["break cost", "4.2", "maybank", "refinance", "420000", "q4 bonus"]
    for question in ISOLATION_PROBES:
        hits = search_episodes(
            question,
            "client_b",
            n_results=TOP_K,
            path=TEST_CHROMA_PATH,
        )
        leaked = any(
            h["client_id"] != "client_b" or _hit_matches(h["text"], a_markers)
            for h in hits
        )
        # Also hard-check metadata client_id
        if any(h["client_id"] != "client_b" for h in hits):
            leaked = True
        status = "LEAK" if leaked else "PASS"
        if leaked:
            leaks += 1
        print(f"[{status}] {question[:70]}")

    print_subheader("ISOLATION RESULT")
    print(f"leaks: {leaks}/{len(ISOLATION_PROBES)}")
    print("isolation: PASS" if leaks == 0 else "isolation: FAIL")

    print_header("DONE")
    print(f"Headline metric → top-3 recall = {recall:.1f}% on {total} probes")
    print(f"Isolation → {leaks} leaks (target 0)")
    print("Note: this is Chroma retrieval only, not full agent answer quality.")

    if leaks > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
