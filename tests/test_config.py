"""Central settings (src/config.py): one place that reads the environment."""
import pytest

from src import config

LLM_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_EXTRACT_MODEL", "LLM_TEMPERATURE",
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_EXTRACT_MODEL", "OPENAI_TEMPERATURE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)


def test_llm_defaults_to_unconfigured_openai():
    s = config.llm_settings()
    assert s.api_key == "" and s.base_url is None
    assert s.model == "gpt-4o-mini" and s.extract_model == "gpt-4o-mini"
    assert not s.configured


def test_old_openai_names_still_work(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
    s = config.llm_settings()
    assert s.api_key == "sk-old" and s.model == "gpt-x" and s.configured


def test_new_llm_names_win_over_openai_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old")
    monkeypatch.setenv("LLM_API_KEY", "sk-new")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_EXTRACT_MODEL", "small")
    s = config.llm_settings()
    assert (s.api_key, s.model, s.extract_model) == ("sk-new", "deepseek-chat", "small")


def test_local_server_needs_no_key(monkeypatch):
    """Ollama / vLLM: an OpenAI-compatible base URL without a key is a working setup."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    s = config.llm_settings()
    assert s.configured and s.base_url == "http://localhost:11434/v1"
    assert s.client_key


def test_switches(monkeypatch):
    monkeypatch.setenv("TAX_AGENT_LLM", "OFF")
    monkeypatch.delenv("TAX_AGENT_RETRIEVAL", raising=False)
    s = config.app_settings()
    assert s.llm_enabled is False and s.retrieval_enabled is True


def test_public_mode_accounts_and_invites_parse(monkeypatch):
    monkeypatch.setenv("TAX_AGENT_MODE", "public")
    monkeypatch.setenv("TAX_AGENT_PREPARERS", "alice:hash-a, bob:pbkdf2_sha256$1$salt$digest ,")
    monkeypatch.setenv("TAX_AGENT_INVITE_CODES", " demo-1 ,demo-2,, ")
    s = config.app_settings()
    assert s.public
    assert s.preparers == {"alice": "hash-a", "bob": "pbkdf2_sha256$1$salt$digest"}
    assert s.invite_codes == ("demo-1", "demo-2")


def test_local_mode_is_the_default(monkeypatch):
    monkeypatch.delenv("TAX_AGENT_MODE", raising=False)
    assert config.app_settings().public is False


def test_bad_account_entry_is_an_error(monkeypatch):
    monkeypatch.setenv("TAX_AGENT_PREPARERS", "no-hash-here")
    with pytest.raises(ValueError, match="TAX_AGENT_PREPARERS"):
        config.app_settings()


def test_limits_parse_and_reject_bad_numbers(monkeypatch):
    monkeypatch.setenv("TAX_AGENT_LLM_CALLS_PER_VISITOR", "50")
    monkeypatch.delenv("TAX_AGENT_LLM_CALLS_PER_DAY", raising=False)
    monkeypatch.setenv("TAX_AGENT_RETENTION_DAYS", "30")
    s = config.app_settings()
    assert (s.llm_calls_per_visitor, s.llm_calls_per_day, s.retention_days) == (50, 2000, 30)
    monkeypatch.setenv("TAX_AGENT_RETENTION_DAYS", "a month")
    with pytest.raises(ValueError, match="TAX_AGENT_RETENTION_DAYS"):
        config.app_settings()
    monkeypatch.setenv("TAX_AGENT_RETENTION_DAYS", "-1")
    with pytest.raises(ValueError, match="negative"):
        config.app_settings()


def test_chat_json_sends_base_url_and_extract_model(monkeypatch):
    """client.py builds the SDK client from these settings (no network: the SDK is faked)."""
    import sys
    import types
    from src.llm import client

    seen = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            seen["model"] = kwargs["model"]
            msg = types.SimpleNamespace(content='{"value": "yes"}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_EXTRACT_MODEL", "qwen3:4b")
    assert client.chat_json("sys", "user") == {"value": "yes"}
    assert seen["client"]["base_url"] == "http://localhost:11434/v1"
    assert seen["client"]["api_key"] == "not-needed"
    assert seen["model"] == "qwen3:4b"
