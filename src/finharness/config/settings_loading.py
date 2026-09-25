"""配置加载：文件读取、深合并、``FINH_`` 环境变量白名单与路径解析。

纯函数集合，只依赖 ``settings_models`` 的模型与类型。``Settings.from_file`` 依次
调用它们把外部输入整理成一个可校验的字典。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from finharness.config.settings_models import (
    CacheKind,
    PermissionMode,
    ProviderKind,
    ProviderSettings,
    SettingsError,
)


def _read_json_object(file_path: Path) -> dict[str, Any]:
    """读取并解析 JSON object；文件不存在返回空 dict，非法 JSON 抛出 SettingsError。"""

    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"settings.json 不是合法 JSON：{file_path}") from exc
    except OSError as exc:
        raise SettingsError(f"无法读取配置文件：{file_path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise SettingsError("settings.json 根节点必须是 JSON object")
    return payload


def _to_mutable(value: Any) -> Any:
    """把冻结默认模型转换为可深合并的普通 Python 容器。"""

    if isinstance(value, BaseModel):
        return {
            name: _to_mutable(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {key: _to_mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_mutable(item) for item in value]
    return deepcopy(value)


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，overrides 优先；不修改入参。"""

    result = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """按路径逐段创建或复用嵌套字典，并写入最终值。"""

    target = payload
    for segment in path[:-1]:
        child = target.get(segment)
        if not isinstance(child, dict):
            child = {}
            target[segment] = child
        target = child
    target[path[-1]] = value


def _is_raw_string(annotation: Any) -> bool:
    if annotation in {str, Path}:
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return all(isinstance(value, str) for value in get_args(annotation))
    return False


def _parse_environment_value(name: str, raw: str, annotation: Any) -> Any:
    """将环境变量的原始字符串解析为注解声明的类型；复杂值要求合法 JSON。"""

    try:
        if _is_raw_string(annotation):
            return raw
        if get_origin(annotation) in {type(None), Literal}:
            return raw
        return TypeAdapter(annotation, config=ConfigDict(strict=True)).validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise SettingsError(
            f"环境变量 {name} 的值无效；复杂值必须使用合法 JSON"
        ) from exc


def _apply_environment_overrides(payload: dict[str, Any]) -> None:
    """把白名单中的 ``FINH_`` 环境变量解析后覆盖写入配置字典。"""

    prefixed = {name.upper(): value for name, value in os.environ.items() if name.upper().startswith("FINH_")}
    for name in prefixed:
        if "__" in name:
            raise SettingsError(f"环境变量 {name} 不支持双下划线，请使用单下划线格式")
        if name not in _ENV_FIELDS:
            raise SettingsError(f"未知环境变量 {name}；请检查名称或移除它")

    for name, (path, annotation) in _ENV_FIELDS.items():
        if name not in prefixed:
            continue
        value = _parse_environment_value(name, prefixed[name], annotation)
        _set_nested(payload, path, value)


def _path_within(child: Path, root: Path) -> bool:
    """``child`` 是否等于 ``root`` 或位于其下（两者均解析后再比较）。

    解析失败（非法字符、过长路径等）时返回 False：宁可放过一次可疑布局，
    也不要因为一个解析异常让配置加载整体失败。
    """
    try:
        resolved_child = Path(child).resolve()
        resolved_root = Path(root).resolve()
    except (OSError, ValueError):
        return False
    return resolved_child == resolved_root or resolved_child.is_relative_to(resolved_root)


def _resolve_path_value(value: Any, base: Path) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_paths(payload: dict[str, Any], base: Path) -> None:
    """将配置中的相对路径按 ``base`` 解析为绝对路径。"""

    for path in (
        ("data", "cache_dir"),
        ("audit", "log_path"),
        ("observability", "logging", "path"),
        ("observability", "trace_store", "db_path"),
        ("server", "static_dir"),
        ("paths", "output_dir"),
        ("paths", "state_dir"),
        ("paths", "memory_db"),
        ("paths", "auth_db"),
        ("paths", "config_db"),
        ("paths", "secret_key"),
        ("paths", "usage_db"),
        ("paths", "compute_jobs_db"),
        ("paths", "compute_packages_dir"),
        ("paths", "skills_dir"),
    ):
        target = payload
        for segment in path[:-1]:
            child = target.get(segment)
            if not isinstance(child, dict):
                break
            target = child
        else:
            if path[-1] in target:
                target[path[-1]] = _resolve_path_value(target[path[-1]], base)


def _format_validation_error(exc: ValidationError) -> str:
    """把 pydantic 校验错误整理为单行可读信息。

    模型级校验（``@model_validator``）抛出的 ``SettingsError`` 会被 pydantic 包成
    ``value_error``，其 `msg` 前缀是无信息量的 "Value error, "；剥掉它，使状态
    布局这类跨字段错误读起来与其它配置错误一致。
    """

    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        message = str(error["msg"])
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]
        details.append(f"{location}: {message}" if location else message)
    return "配置校验失败：" + "; ".join(details)


_ENV_FIELDS: dict[str, tuple[tuple[str, ...], Any]] = {
    "FINH_PROVIDERS": (("providers",), dict[str, ProviderSettings]),
    "FINH_MODEL_PROVIDER": (("model", "provider"), str),
    "FINH_MODEL_MODEL_NAME": (("model", "model_name"), str),
    "FINH_MODEL_TEMPERATURE": (("model", "temperature"), float),
    "FINH_MODEL_MAX_TOKENS": (("model", "max_tokens"), int),
    "FINH_MODEL_THINKING_ENABLED": (("model", "thinking", "enabled"), bool),
    "FINH_MODEL_THINKING_BUDGET_TOKENS": (("model", "thinking", "budget_tokens"), int),
    "FINH_PERMISSION_DEFAULT_MODE": (("permission", "default_mode"), PermissionMode),
    "FINH_TOOLS_TIMEOUT_DEFAULT_S": (("tools", "timeout_default_s"), int),
    "FINH_TOOLS_TIMEOUT_OVERRIDES": (("tools", "timeout_overrides"), dict[str, int]),
    "FINH_TOOLS_RESULT_TOKEN_OVERRIDES": (("tools", "result_token_overrides"), dict[str, int]),
    "FINH_TOOLS_RESIDENT": (("tools", "resident"), list[str]),
    "FINH_TOOLS_LAZY": (("tools", "lazy"), list[str]),
    "FINH_DATA_ADAPTER_ORDER": (("data", "adapter_order"), list[str]),
    "FINH_DATA_TUSHARE_TOKEN_ENV": (("data", "tushare_token_env"), str),
    "FINH_DATA_THROTTLE_SECONDS": (("data", "throttle_seconds"), float),
    "FINH_DATA_CACHE_DIR": (("data", "cache_dir"), Path),
    "FINH_DATA_CACHE_TTL_DAYS": (("data", "cache_ttl_days"), dict[CacheKind, int]),
    "FINH_CONTEXT_MAX_TURNS": (("context", "max_turns"), int),
    "FINH_CONTEXT_COMPACTION_RATIO": (("context", "compaction_ratio"), float),
    "FINH_CONTEXT_CONTEXT_WINDOW_TOKENS": (("context", "context_window_tokens"), int),
    "FINH_CONTEXT_TRIM_ROWS": (("context", "trim_rows"), int),
    "FINH_CONTEXT_MAX_RESULT_TOKENS": (("context", "max_result_tokens"), int),
    "FINH_CONTEXT_COMPACTION_RESULT_TOKENS": (("context", "compaction_result_tokens"), int),
    "FINH_CONTEXT_MAX_TOOL_SCHEMA_TOKENS": (("context", "max_tool_schema_tokens"), int),
    "FINH_CONTEXT_MAX_IDENTICAL_TOOL_CALLS": (("context", "max_identical_tool_calls"), int),
    "FINH_CONTEXT_PLAN_STALL_TURNS": (("context", "plan_stall_turns"), int),
    "FINH_CONTEXT_SUMMARY_BUDGET_RATIO": (("context", "summary_budget_ratio"), float),
    "FINH_CONTEXT_MIN_RECENT_ROUNDS": (("context", "min_recent_rounds"), int),
    "FINH_CONTEXT_RETENTION_CONVERSATIONS": (("context", "retention_conversations"), int),
    "FINH_CONTEXT_RETENTION_DAYS": (("context", "retention_days"), int),
    "FINH_CONTEXT_SHORT_MEM_CAP": (("context", "short_mem_cap"), int),
    "FINH_CONTEXT_RECALL_MAX_TOKENS": (("context", "recall_max_tokens"), int),
    "FINH_CONTEXT_SUMMARY_INJECT_MAX_TOKENS": (("context", "summary_inject_max_tokens"), int),
    "FINH_LTM_DISTILL_IDLE_S": (("ltm", "distill_idle_s"), int),
    "FINH_LTM_DISTILL_BATCH": (("ltm", "distill_batch"), int),
    "FINH_LTM_DISTILL_MAX_ATTEMPTS": (("ltm", "distill_max_attempts"), int),
    "FINH_LTM_RECENT_EPISODES": (("ltm", "recent_episodes"), int),
    "FINH_LTM_INJECT_MAX_TOKENS": (("ltm", "inject_max_tokens"), int),
    "FINH_LTM_RECALL_MAX_TOKENS": (("ltm", "recall_max_tokens"), int),
    "FINH_LTM_RETENTION_EPISODES": (("ltm", "retention_episodes"), int),
    "FINH_LTM_RETENTION_DAYS": (("ltm", "retention_days"), int),
    "FINH_LTM_DISTILL_SEMANTICS": (("ltm", "distill_semantics"), bool),
    "FINH_LTM_AUTO_TASK_EPISODES": (("ltm", "auto_task_episodes"), bool),
    "FINH_LTM_RETENTION_FACTS": (("ltm", "retention_facts"), int),
    "FINH_LTM_RETENTION_FACTS_DAYS": (("ltm", "retention_facts_days"), int),
    "FINH_LTM_SEMANTIC_TOP_K": (("ltm", "semantic_top_k"), int),
    "FINH_LTM_SEMANTIC_INJECT_MAX_TOKENS": (("ltm", "semantic_inject_max_tokens"), int),
    "FINH_LTM_EMBEDDINGS_BASE_URL": (("ltm", "embeddings", "base_url"), str),
    "FINH_LTM_EMBEDDINGS_ENV_KEY": (("ltm", "embeddings", "env_key"), str),
    "FINH_LTM_EMBEDDINGS_MODEL_NAME": (("ltm", "embeddings", "model_name"), str),
    "FINH_LTM_EMBEDDINGS_TIMEOUT_S": (("ltm", "embeddings", "timeout_s"), float),
    "FINH_LTM_VECTOR_DB_KIND": (("ltm", "vector_db", "kind"), Literal["qdrant"]),
    "FINH_LTM_VECTOR_DB_URL": (("ltm", "vector_db", "url"), str),
    "FINH_LTM_VECTOR_DB_COLLECTION": (("ltm", "vector_db", "collection"), str),
    "FINH_LTM_VECTOR_DB_API_KEY_ENV": (("ltm", "vector_db", "api_key_env"), str),
    "FINH_LTM_VECTOR_DB_TIMEOUT_S": (("ltm", "vector_db", "timeout_s"), float),
    "FINH_LTM_VECTOR_DB_DIM": (("ltm", "vector_db", "dim"), int),
    "FINH_AUDIT_LOG_PATH": (("audit", "log_path"), Path),
    "FINH_OBSERVABILITY_LOGGING_LEVEL": (
        ("observability", "logging", "level"),
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
    ),
    "FINH_OBSERVABILITY_LOGGING_JSON_FORMAT": (("observability", "logging", "json_format"), bool),
    "FINH_OBSERVABILITY_LOGGING_PATH": (("observability", "logging", "path"), Path),
    "FINH_OBSERVABILITY_LOGGING_CAPTURE_PAYLOADS": (
        ("observability", "logging", "capture_payloads"),
        bool,
    ),
    "FINH_OBSERVABILITY_LOGGING_MAX_PAYLOAD_CHARS": (
        ("observability", "logging", "max_payload_chars"),
        int,
    ),
    "FINH_OBSERVABILITY_METRICS_ENABLED": (("observability", "metrics", "enabled"), bool),
    "FINH_OBSERVABILITY_TRACING_ENABLED": (("observability", "tracing", "enabled"), bool),
    "FINH_OBSERVABILITY_TRACING_BACKEND": (
        ("observability", "tracing", "backend"),
        Literal["langsmith"],
    ),
    "FINH_OBSERVABILITY_TRACING_PROJECT": (("observability", "tracing", "project"), str),
    "FINH_OBSERVABILITY_TRACING_ENV_KEY": (("observability", "tracing", "env_key"), str),
    "FINH_OBSERVABILITY_TRACING_CAPTURE_PAYLOADS": (
        ("observability", "tracing", "capture_payloads"),
        bool,
    ),
    "FINH_OBSERVABILITY_TRACE_STORE_ENABLED": (
        ("observability", "trace_store", "enabled"),
        bool,
    ),
    "FINH_OBSERVABILITY_TRACE_STORE_DB_PATH": (
        ("observability", "trace_store", "db_path"),
        Path,
    ),
    "FINH_OBSERVABILITY_TRACE_STORE_CAPTURE_PAYLOADS": (
        ("observability", "trace_store", "capture_payloads"),
        bool,
    ),
    "FINH_OBSERVABILITY_TRACE_STORE_RETENTION_DAYS": (
        ("observability", "trace_store", "retention_days"),
        int,
    ),
    "FINH_SERVER_HOST": (("server", "host"), str),
    "FINH_SERVER_PORT": (("server", "port"), int),
    "FINH_SERVER_SESSION_TTL_S": (("server", "session_ttl_s"), int),
    "FINH_SERVER_CONFIRM_TTL_S": (("server", "confirm_ttl_s"), int),
    "FINH_SERVER_STATIC_DIR": (("server", "static_dir"), Path),
    "FINH_SERVER_ALLOW_REMOTE": (("server", "allow_remote"), bool),
    "FINH_SERVER_METRICS_TOKEN": (("server", "metrics_token"), str),
    "FINH_COMPUTE_REMOTE_WORKER_URL": (("compute", "remote_worker_url"), str),
    "FINH_COMPUTE_HMAC_SECRET_ENV": (("compute", "hmac_secret_env"), str),
    "FINH_COMPUTE_LEASE_SECONDS": (("compute", "lease_seconds"), int),
    "FINH_COMPUTE_MAX_WAITING_PER_USER": (("compute", "max_waiting_per_user"), int),
    "FINH_COMPUTE_MAX_ATTEMPTS": (("compute", "max_attempts"), int),
    "FINH_AUTH_TOKEN_TTL_S": (("auth", "token_ttl_s"), int),
    "FINH_AUTH_MIN_PASSWORD_LEN": (("auth", "min_password_len"), int),
    "FINH_AUTH_SECURE_COOKIE": (("auth", "secure_cookie"), bool),
    "FINH_AUTH_ALLOW_REGISTER": (("auth", "allow_register"), bool),
    "FINH_AUTH_LOGIN_MAX_ATTEMPTS": (("auth", "login_max_attempts"), int),
    "FINH_AUTH_LOGIN_WINDOW_S": (("auth", "login_window_s"), int),
    "FINH_AUTH_REGISTER_MAX_ATTEMPTS": (("auth", "register_max_attempts"), int),
    "FINH_AUTH_REGISTER_WINDOW_S": (("auth", "register_window_s"), int),
    "FINH_AUTH_CLAIM_LEGACY_ON_FIRST_REGISTER": (
        ("auth", "claim_legacy_on_first_register"),
        bool,
    ),
    "FINH_AUTH_ADMIN_BOOTSTRAP": (("auth", "admin_bootstrap"), bool),
    "FINH_QUOTA_TURNS_PER_WINDOW": (("quota", "turns_per_window"), int),
    "FINH_QUOTA_WINDOW_S": (("quota", "window_s"), int),
    "FINH_QUOTA_MAX_CONCURRENT_STREAMS": (("quota", "max_concurrent_streams"), int),
    "FINH_PATHS_OUTPUT_DIR": (("paths", "output_dir"), Path),
    "FINH_PATHS_STATE_DIR": (("paths", "state_dir"), Path),
    "FINH_PATHS_MEMORY_DB": (("paths", "memory_db"), Path),
    "FINH_PATHS_AUTH_DB": (("paths", "auth_db"), Path),
    "FINH_PATHS_CONFIG_DB": (("paths", "config_db"), Path),
    "FINH_PATHS_SECRET_KEY": (("paths", "secret_key"), Path),
    "FINH_PATHS_USAGE_DB": (("paths", "usage_db"), Path),
    "FINH_PATHS_COMPUTE_JOBS_DB": (("paths", "compute_jobs_db"), Path),
    "FINH_PATHS_COMPUTE_PACKAGES_DIR": (("paths", "compute_packages_dir"), Path),
    "FINH_PATHS_SKILLS_DIR": (("paths", "skills_dir"), Path),
    "FINH_SEARCH_KIND": (("search", "kind"), Literal["tavily"]),
    "FINH_SEARCH_BASE_URL": (("search", "base_url"), str),
    "FINH_SEARCH_ENV_KEY": (("search", "env_key"), str),
    "FINH_SEARCH_TIMEOUT_S": (("search", "timeout_s"), float),
    "FINH_SEARCH_API_KEY": (("search", "api_key"), str),
    "FINH_SEARCH_PROXY": (("search", "proxy"), str),
    "FINH_SEARCH_LOCAL_PDF_FALLBACK": (("search", "local_pdf_fallback"), bool),
    "FINH_FUYAO_ENABLED": (("fuyao", "enabled"), bool),
    "FINH_FUYAO_KIND": (("fuyao", "kind"), Literal["mcp"]),
    "FINH_FUYAO_BASE_URL": (("fuyao", "base_url"), str),
    "FINH_FUYAO_ENV_KEY": (("fuyao", "env_key"), str),
    "FINH_FUYAO_TIMEOUT_S": (("fuyao", "timeout_s"), float),
    "FINH_FUYAO_API_KEY": (("fuyao", "api_key"), str),
    "FINH_FUYAO_PROXY": (("fuyao", "proxy"), str),
    "FINH_FUYAO_CATALOG_TTL_S": (("fuyao", "catalog_ttl_s"), float),
}


for _provider_name in ("DEEPSEEK", "KIMI", "GLM", "VOLCANO", "QWEN", "FAKE"):
    _provider_key = _provider_name.lower()
    _ENV_FIELDS.update(
        {
            f"FINH_PROVIDERS_{_provider_name}_KIND": (
                ("providers", _provider_key, "kind"),
                ProviderKind,
            ),
            f"FINH_PROVIDERS_{_provider_name}_BASE_URL": (
                ("providers", _provider_key, "base_url"),
                str,
            ),
            f"FINH_PROVIDERS_{_provider_name}_ENV_KEY": (
                ("providers", _provider_key, "env_key"),
                str,
            ),
            f"FINH_PROVIDERS_{_provider_name}_API_VERSION": (
                ("providers", _provider_key, "api_version"),
                str,
            ),
            f"FINH_PROVIDERS_{_provider_name}_FIRST_BYTE_TIMEOUT_S": (
                ("providers", _provider_key, "first_byte_timeout_s"),
                float,
            ),
            f"FINH_PROVIDERS_{_provider_name}_IDLE_TIMEOUT_S": (
                ("providers", _provider_key, "idle_timeout_s"),
                float,
            ),
        }
    )
