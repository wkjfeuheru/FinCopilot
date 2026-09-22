#!/bin/sh
# Railway 容器入口：以 root 接管 Volume，降权后以 app 用户运行 Uvicorn。
#
# 为什么以 root 启动（RAILWAY_RUN_UID=0 预检决议）：Railway 的 Volume 以
# root 挂载，容器内非 root 用户无法在首次部署时创建子目录。这里 root 只做
# 一件事——把 Volume 属主改为镜像内的 app 用户（UID 10001），然后立即用
# gosu 降权；Uvicorn 进程本身绝不以 root 运行。
#
# 所有持久化路径都从 RAILWAY_VOLUME_MOUNT_PATH 派生并以 FINH_* 注入：
# 环境变量优先于镜像内 settings.json，因此无论 Volume 挂在哪个路径，
# 状态/缓存/产物/日志都整体跟随挂载点，重部署不会漂移。
set -eu

DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"
APP_USER="${FINH_CONTAINER_USER:-app}"

mkdir -p "$DATA_DIR/logs"
chown -R "$APP_USER" "$DATA_DIR"

# 预创建审计日志父目录：Settings.validate_runtime 要求其存在且可写，
# 而各 SQLite store 的目录是懒创建的——唯独日志目录必须在启动前就位。
# 其余子目录由 store 在运行时自行创建，这里不预先铺开。

export FINH_PATHS_STATE_DIR="$DATA_DIR/state"
export FINH_PATHS_MEMORY_DB="$DATA_DIR/state/memory.db"
export FINH_PATHS_AUTH_DB="$DATA_DIR/state/users.db"
export FINH_PATHS_CONFIG_DB="$DATA_DIR/state/config.db"
export FINH_PATHS_SECRET_KEY="$DATA_DIR/state/secret.key"
export FINH_PATHS_OUTPUT_DIR="$DATA_DIR/output"
export FINH_DATA_CACHE_DIR="$DATA_DIR/data_cache"
export FINH_AUDIT_LOG_PATH="$DATA_DIR/logs/audit.jsonl"

exec gosu "$APP_USER" uvicorn finharness.server.api:create_production_app \
    --factory --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
