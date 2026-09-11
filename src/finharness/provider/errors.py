"""Provider error taxonomy and retry metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite


class ProviderError(RuntimeError):
    """A provider failure with retry information for the engine."""

    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = self.default_retryable if retryable is None else retryable
        self.retry_after_s = retry_after_s


class AuthError(ProviderError):
    default_retryable = False


class RateLimitError(ProviderError):
    default_retryable = True


class ServerError(ProviderError):
    default_retryable = True


class NetworkError(ProviderError):
    default_retryable = False


def parse_retry_after(value: str | None) -> float | None:
    """Parse HTTP ``Retry-After`` seconds or date without raising on bad input."""
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return delay if delay >= 0 and isfinite(delay) else None
