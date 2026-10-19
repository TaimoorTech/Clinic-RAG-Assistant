"""Deterministic safety checks that run BEFORE any LLM call."""
import re

DISCLAIMER = ("This is general information from the clinic's documents, not medical advice. "
              "Please speak to a clinician about your own health.")

NO_ANSWER = ("I'm sorry, I couldn't find that in the clinic's information. "
             "Please contact reception on 020 5550 0142 during opening hours.")

EMERGENCY_REPLY = ("This sounds like it could be an emergency. Please call 999 now or go to "
                   "your nearest emergency department. This clinic does not provide "
                   "emergency care.")

# Deliberately simple and conservative; errs on the side of redirecting.
_EMERGENCY = re.compile(
    r"\b(chest pain|can'?t breathe|cannot breathe|difficulty breathing|trouble breathing|"
    r"severe bleeding|heavy bleeding|unconscious|unresponsive|stroke|heart attack|"
    r"overdose|seizure|choking|anaphyla\w*)\b",
    re.IGNORECASE,
)


def is_emergency(text: str) -> bool:
    return bool(_EMERGENCY.search(text))