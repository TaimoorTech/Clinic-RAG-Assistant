from app.guardrails import is_emergency


def test_emergency_phrases_detected():
    assert is_emergency("I have chest pain, can I book an appointment?")
    assert is_emergency("My dad can't breathe")
    assert is_emergency("possible OVERDOSE")


def test_normal_questions_not_flagged():
    assert not is_emergency("Do I need to fast before a blood test?")
    assert not is_emergency("What time do you close on Saturday?")
