import pytest

from app.llm import ConfigError, LLMError, generate, get_llm


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.seen = content, exc, None

    def invoke(self, messages):
        self.seen = messages
        if self.exc:
            raise self.exc
        return FakeResp(self.content)


def test_generate_returns_stripped_text_and_sends_both_messages():
    fake = FakeLLM("  hello  ")
    assert generate("SYS", "USER", llm=fake) == "hello"
    assert fake.seen == [("system", "SYS"), ("human", "USER")]


def test_generate_handles_list_content():
    assert generate("s", "u", llm=FakeLLM([{"type": "text", "text": "hi"}])) == "hi"


def test_generate_wraps_provider_errors():
    with pytest.raises(LLMError) as e:
        generate("s", "u", llm=FakeLLM(exc=RuntimeError("quota exceeded")))
    assert "quota exceeded" in str(e.value)


def test_generate_rejects_empty_answer():
    with pytest.raises(LLMError):
        generate("s", "u", llm=FakeLLM("   "))


def test_get_llm_reports_missing_settings():
    with pytest.raises(ConfigError) as e:
        get_llm(env={})
    assert "OPENAI_API_KEY" in str(e.value) and "OPENAI_MODEL" in str(e.value)