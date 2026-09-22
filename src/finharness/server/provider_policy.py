"""用户 Provider 地址在本地与远程部署下的统一安全策略。"""

from __future__ import annotations

from urllib.parse import urlparse

from finharness.config.settings import Settings


def validate_user_provider_url(
    base_url: str | None,
    kind: str,
    settings: Settings,
) -> str | None:
    """校验用户提供的 Provider 地址；通过返回 ``None``，否则返回字段错误。"""

    if kind == "fake":
        return None
    if not base_url or not base_url.strip():
        return "base_url 不能为空"

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "base_url 必须是绝对 HTTP(S) URL"
    if not settings.server.allow_remote:
        return None
    if parsed.scheme != "https":
        return "远程部署的 Provider 地址必须使用 HTTPS"

    normalized_url = base_url.rstrip("/")
    preset_urls = {
        preset.base_url.rstrip("/")
        for preset in settings.providers.values()
        if preset.base_url is not None
    }
    if normalized_url not in preset_urls:
        return "远程部署只允许使用运维预设的 Provider 地址"
    return None
