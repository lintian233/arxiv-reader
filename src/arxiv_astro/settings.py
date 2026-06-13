from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir


DEBUG = False
APP_NAME = "arxiv-reader"


def parse_bool(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = enabled


def debug_log(message: str, **fields: Any) -> None:
    if not DEBUG:
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[DEBUG] {message}{suffix}", file=__import__("sys").stderr)


def default_paper_data_dir() -> str:
    return user_data_dir(APP_NAME, appauthor=False)


def default_data_dir() -> str:
    return default_paper_data_dir()


def default_runs_dir() -> str:
    return "."


def default_config_dir() -> str:
    return user_config_dir(APP_NAME, appauthor=False)


def default_env_path() -> Path:
    return Path(default_config_dir()) / ".env"


def optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    selection_model: str = ""
    interpretation_model: str = ""
    paper_data_dir: str = ""
    runs_dir: str = "."
    request_timeout: float = 30.0
    llm_request_timeout: float = 180.0
    max_input_chars: int = 400000
    llm_max_output_tokens: int = 24000
    paper_interests: str = ""
    fetch_results: int | None = None
    selection_max_input_chars: int = 220000
    selection_summary_max_chars: int = 2400
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", cls.base_url),
            model=os.getenv("DEEPSEEK_MODEL", cls.model),
            selection_model=os.getenv("DEEPSEEK_SELECTION_MODEL", os.getenv("DEEPSEEK_MODEL", cls.model)),
            interpretation_model=os.getenv("DEEPSEEK_INTERPRETATION_MODEL", os.getenv("DEEPSEEK_MODEL", cls.model)),
            paper_data_dir=os.getenv("PAPER_DATA_DIR", os.getenv("OUTPUT_DIR", default_paper_data_dir())),
            runs_dir=os.getenv("RUNS_DIR", default_runs_dir()),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", cls.request_timeout)),
            llm_request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", cls.llm_request_timeout)),
            max_input_chars=int(os.getenv("MAX_INPUT_CHARS", cls.max_input_chars)),
            llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", cls.llm_max_output_tokens)),
            paper_interests=os.getenv("PAPER_INTERESTS", cls.paper_interests),
            fetch_results=optional_int(os.getenv("FETCH_RESULTS")),
            selection_max_input_chars=int(os.getenv("SELECTION_MAX_INPUT_CHARS", cls.selection_max_input_chars)),
            selection_summary_max_chars=int(os.getenv("SELECTION_SUMMARY_MAX_CHARS", cls.selection_summary_max_chars)),
            debug=parse_bool(os.getenv("DEBUG")),
        )
        set_debug(settings.debug)
        return settings

    @property
    def output_dir(self) -> str:
        return self.paper_data_dir
