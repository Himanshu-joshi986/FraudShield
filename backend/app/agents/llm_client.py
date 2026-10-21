"""
Phase 9 -- LLM client (Gemini, free tier).

VERIFIED: this sandbox's network egress returns 403 Forbidden for
generativelanguage.googleapis.com (checked directly, not assumed). So no
live Gemini call was made in the build environment. This client is
written against the real `google-generativeai` API and works once you
run it with your own key and normal internet access.

Design constraints from the free tier (as low as 10 RPM / 500 RPD on
some models -- see the conversation this was scoped from): a hard
per-day request budget and per-minute rate limiting are enforced HERE,
in the client, not left to the caller to remember. Every agent that uses
this client gets the budget for free.

Section 57 (error handling): if the LLM is unavailable, unconfigured, or
the budget is exhausted, `complete()` returns a DEGRADED result rather
than raising -- callers (the agents) use this to fall back to a
deterministic template rather than crashing the pipeline.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    text: str
    succeeded: bool
    degraded_reason: str | None = None
    provider: str = "gemini"
    model: str = ""


class RequestBudget:
    """Tracks requests-per-minute and requests-per-day, entirely in-process
    (no external state) -- resets on process restart, which is acceptable
    for a demo/dev budget guard, not a production rate limiter."""

    def __init__(self, max_per_minute: int, max_per_day: int):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self._minute_window: deque[float] = deque()
        self._day_count = 0
        self._day_start = time.time()

    def _prune(self):
        now = time.time()
        while self._minute_window and now - self._minute_window[0] > 60:
            self._minute_window.popleft()
        if now - self._day_start > 86400:
            self._day_count = 0
            self._day_start = now

    def can_proceed(self) -> tuple[bool, str | None]:
        self._prune()
        if self._day_count >= self.max_per_day:
            return False, f"daily request budget exhausted ({self.max_per_day}/day)"
        if len(self._minute_window) >= self.max_per_minute:
            return False, f"per-minute rate limit reached ({self.max_per_minute}/min) -- retry shortly"
        return True, None

    def record(self):
        self._minute_window.append(time.time())
        self._day_count += 1


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                max_per_minute: int | None = None, max_per_day: int | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "gemini-1.5-flash")
        self.budget = RequestBudget(
            max_per_minute=max_per_minute or int(os.environ.get("LLM_MAX_REQUESTS_PER_MINUTE", 10)),
            max_per_day=max_per_day or int(os.environ.get("LLM_MAX_REQUESTS_PER_DAY", 400)),
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def complete(self, prompt: str, max_retries: int = 3) -> LLMResult:
        if not self.api_key:
            return LLMResult(text="", succeeded=False,
                            degraded_reason="GEMINI_API_KEY not configured", model=self.model)

        can_proceed, reason = self.budget.can_proceed()
        if not can_proceed:
            return LLMResult(text="", succeeded=False, degraded_reason=reason, model=self.model)

        try:
            client = self._get_client()
        except ImportError:
            return LLMResult(text="", succeeded=False,
                            degraded_reason="google-generativeai not installed "
                                             "(pip install google-generativeai)", model=self.model)

        backoff = 1.0
        last_error = None
        for attempt in range(max_retries):
            try:
                self.budget.record()
                response = client.generate_content(prompt)
                return LLMResult(text=response.text, succeeded=True, model=self.model)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2

        return LLMResult(text="", succeeded=False,
                        degraded_reason=f"LLM call failed after {max_retries} attempts: {last_error}",
                        model=self.model)


_client_instance: GeminiClient | None = None


def get_llm_client() -> GeminiClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance
