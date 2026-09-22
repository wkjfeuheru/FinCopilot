# Railway 生产镜像：单 service、单 replica、单 Uvicorn worker（设计文档约束）。
# 布局注解：
#   /app/src/...        editable 安装的包源码；api.py 按 parents[3] 相对解析
#                       frontend/dist，因此 /app 必须保留仓库根布局
#   /app/settings.json  无密钥的远程模式配置；密钥只经 FINH_*/Provider 环境变量注入
#   /data               Railway Volume 挂载点（root 属主），entrypoint 降权前接管
#
# 环境变量以 FINH_ 白名单为唯一入口，未知 FINH_* 变量会在启动时被拒绝
# （config/settings.py _apply_environment_overrides）。

# -- 前端：npm 锁文件安装 + 构建 ----------------------------------------------
FROM node:24-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
# 只需要构建输入；node_modules/dist 均在 .dockerignore 中排除
COPY frontend/ ./
RUN npm run build

# -- Python 依赖：锁文件同步（含 observability extra） --------------------------
# uv 经 pip 安装（ghcr.io 在部分网络不可达；pypi 覆盖面更稳）。
# 版本钉死：uv.lock 由 0.11.29 生成，更高版本的 uv 会要求重写锁文件。
# 升级 uv 时需同步运行 uv lock 刷新 uv.lock。
# UV_INDEX_URL 默认官方源；网络受限环境构建时传
# --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple。
# --frozen（而非 --locked）：完全信任锁文件、跳过索引新鲜度校验——
# 索引元数据写在锁文件里，换镜像源会让 --locked 必然失败；锁文件的
# 新鲜度由 uv lock --check 门禁保证（CI，Task 5）。
FROM python:3.13-slim-bookworm AS builder
ARG UV_VERSION=0.11.29
ARG UV_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "$UV_INDEX_URL" "uv==$UV_VERSION"
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN UV_INDEX_URL="$UV_INDEX_URL" uv sync --frozen --no-dev --extra observability

# -- 运行镜像 -------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime
# gosu 只用于 entrypoint 的 root→app 降权；系统依赖为零（纯 Python 栈）。
RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --system --create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=frontend /build/dist /app/frontend/dist
COPY pyproject.toml uv.lock ./
COPY src/ src/
# 无密钥的远程模式配置（路径指向 /data）；密钥经 Railway Variables 注入
COPY settings.railway.example.json /app/settings.json

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# root 启动仅为接管 root 挂载的 /data Volume（RAILWAY_RUN_UID=0 预检决议）；
# chown 后降权到非 root 的 app 用户运行 Uvicorn。
COPY ops/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
