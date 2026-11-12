"""Rotating audit logger and LangChain callback handler for tool and LLM observability."""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_AUDIT_LOG_FILE = DEFAULT_LOG_DIR / "audit.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5


def get_audit_logger(log_file: Path | str = DEFAULT_AUDIT_LOG_FILE) -> logging.Logger:
    """Configure and return a dedicated rotating audit logger."""
    path = Path(log_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"cloudcommerce.audit.{str(path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def log_audit_event(event_type: str, payload: dict[str, Any], log_file: Path | str = DEFAULT_AUDIT_LOG_FILE) -> None:
    """Record a structured JSONL audit record."""
    logger = get_audit_logger(log_file)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z",
        "event": event_type,
        **payload,
    }
    logger.info(json.dumps(record, ensure_ascii=False, default=str))


class AuditCallbackHandler(BaseCallbackHandler):
    """LangChain / LangGraph callback handler tracking tool calls, LLM latency, and tokens."""

    def __init__(
        self,
        session_id: str | None = None,
        customer_id: str | None = None,
        log_file: Path | str = DEFAULT_AUDIT_LOG_FILE,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.customer_id = customer_id
        self.log_file = log_file
        self._tool_starts: dict[UUID, tuple[float, str, Any]] = {}
        self._llm_starts: dict[UUID, float] = {}
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def get_token_usage(self) -> dict[str, int]:
        """Return total aggregated tokens consumed across all model calls in this turn."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        args = inputs if inputs is not None else input_str
        self._tool_starts[run_id] = (time.perf_counter(), tool_name, args)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start_info = self._tool_starts.pop(run_id, None)
        duration_ms = None
        tool_name = "unknown_tool"
        tool_args = None
        if start_info:
            start_time, tool_name, tool_args = start_info
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        output_str = output if isinstance(output, str) else str(output)
        log_audit_event(
            "tool_call",
            {
                "session_id": self.session_id,
                "customer_id": self.customer_id,
                "tool": tool_name,
                "arguments": tool_args,
                "response_time_ms": duration_ms,
                "status": "success",
                "outcome": output_str[:500] if len(output_str) > 500 else output_str,
            },
            log_file=self.log_file,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start_info = self._tool_starts.pop(run_id, None)
        duration_ms = None
        tool_name = "unknown_tool"
        tool_args = None
        if start_info:
            start_time, tool_name, tool_args = start_info
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        log_audit_event(
            "tool_call",
            {
                "session_id": self.session_id,
                "customer_id": self.customer_id,
                "tool": tool_name,
                "arguments": tool_args,
                "response_time_ms": duration_ms,
                "status": "error",
                "error": str(error),
            },
            log_file=self.log_file,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[run_id] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._llm_starts.pop(run_id, None)
        duration_ms = None
        duration_sec = 0.0
        if start_time is not None:
            duration_sec = time.perf_counter() - start_time
            duration_ms = round(duration_sec * 1000, 2)

        p_tok = 0
        c_tok = 0
        t_tok = 0

        # 1. Check response.llm_output for token usage
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            p_tok = usage.get("prompt_tokens") or 0
            c_tok = usage.get("completion_tokens") or 0
            t_tok = usage.get("total_tokens") or (p_tok + c_tok)

        # 2. Check message metadata
        if t_tok == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if isinstance(msg, BaseMessage):
                        if getattr(msg, "usage_metadata", None):
                            meta = msg.usage_metadata or {}
                            p_tok = meta.get("input_tokens") or 0
                            c_tok = meta.get("output_tokens") or 0
                            t_tok = meta.get("total_tokens") or (p_tok + c_tok)
                        elif getattr(msg, "response_metadata", None):
                            meta = msg.response_metadata.get("token_usage") or {}
                            p_tok = meta.get("prompt_tokens") or 0
                            c_tok = meta.get("completion_tokens") or 0
                            t_tok = meta.get("total_tokens") or (p_tok + c_tok)
                    if t_tok > 0:
                        break

        # Accumulate tokens across all LLM steps in this turn
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += t_tok

        speed_tps = round(t_tok / duration_sec, 2) if t_tok > 0 and duration_sec > 0 else None

        log_audit_event(
            "llm_generation",
            {
                "session_id": self.session_id,
                "customer_id": self.customer_id,
                "duration_ms": duration_ms,
                "tokens": {
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": t_tok,
                },
                "speed_tokens_per_sec": speed_tps,
            },
            log_file=self.log_file,
        )
