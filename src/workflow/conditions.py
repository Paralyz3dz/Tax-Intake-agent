"""
conditions.py: the playbook condition format.

    [{"fact": "has_side_income", "equals": true}]    holds only when EVERY item holds

Used for "ask_when" (may this question be asked yet?) and "when" (does this escalation fire?).
Conditions are data, not Python strings, so loading a playbook never runs code (no eval).
"""
from __future__ import annotations


def check(conditions: list[dict] | None, facts: dict) -> bool:
    """No conditions -> True. An unknown (None) fact -> False, so a follow-up is never asked
    before its parent question is answered.
    """
    if not conditions:
        return True
    for cond in conditions:
        actual = facts.get(cond["fact"])
        if actual is None or actual != cond["equals"]:
            return False
    return True
