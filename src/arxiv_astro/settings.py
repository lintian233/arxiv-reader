from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEBUG = False


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


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    output_dir: str = "data"
    pdf_dir: str = "data/pdfs"
    request_timeout: float = 30.0
    llm_request_timeout: float = 180.0
    max_input_chars: int = 20000
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", cls.base_url),
            model=os.getenv("DEEPSEEK_MODEL", cls.model),
            output_dir=os.getenv("OUTPUT_DIR", cls.output_dir),
            pdf_dir=os.getenv("PDF_DIR", cls.pdf_dir),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", cls.request_timeout)),
            llm_request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", cls.llm_request_timeout)),
            max_input_chars=int(os.getenv("MAX_INPUT_CHARS", cls.max_input_chars)),
            debug=parse_bool(os.getenv("DEBUG")),
        )
        set_debug(settings.debug)
        return settings
