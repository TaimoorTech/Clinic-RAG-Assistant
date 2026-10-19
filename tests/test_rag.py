from app.guardrails import DISCLAIMER, NO_ANSWER
from app.rag import answer_question, build_user_prompt

HIT = {"text": "Cancel at least 24 hours in advance.", "source": "appointments_and_cancellation.md",
       "section": "Cancelling", "score": 0.82}


def boom(*_):
    raise AssertionError("should not be called")


def test_no_hits_skips_the_llm():
    out = answer_question("Do you offer physiotherapy?", retriever=lambda q: [], generator=boom)
    assert out["kind"] == "no_context" and out["answer"] == NO_ANSWER and out["sources"] == []


def test_emergency_skips_retrieval_and_llm():
    out = answer_question("I have chest pain", retriever=boom, generator=boom)
    assert out["kind"] == "emergency" and "999" in out["answer"]


def test_normal_answer_has_sources_and_disclaimer():
    seen = {}

    def fake_llm(system, user):
        seen["user"] = user
        return "Cancel 24 hours before [1]."

    out = answer_question("How late can I cancel?", retriever=lambda q: [HIT], generator=fake_llm)
    assert out["kind"] == "answer"
    assert out["answer"].endswith(DISCLAIMER)
    assert out["sources"] == [{"n": 1, "source": HIT["source"], "section": "Cancelling", "score": 0.82}]
    assert "[1]" in seen["user"] and "How late can I cancel?" in seen["user"]


def test_prompt_numbers_passages_and_wraps_question():
    p = build_user_prompt("Q?", [HIT, {**HIT, "section": ""}])
    assert "[1] (source:" in p and "[2] (source:" in p and "general" in p
    assert p.rstrip().endswith("</question>")
