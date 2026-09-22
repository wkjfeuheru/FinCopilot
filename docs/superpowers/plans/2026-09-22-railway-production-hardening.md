# Railway 生产化加固 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 FinHarness 能以 Railway 的单服务、单 worker 形态安全、可重复地上线。

**Architecture:** FastAPI 同源托管 React 产物。Railway 终止 TLS；容器只监听 Railway 提供的 PORT。远程模式下，用户可保存 API Key，但 Provider base_url 必须精确匹配运维在 settings.providers 中声明的 HTTPS 预设；Railway Volume 挂载到 /data 保存状态。

**Tech Stack:** Python 3.13、FastAPI、SQLite、pytest、Ruff、Node/Vite、Docker、Railway。

## Global Constraints

* Railway 只运行一个 service、一个 replica、一个 Uvicorn worker。
* 所有 Railway 持久化路径位于 /data；密钥、数据库和 Prompt/响应正文不进入镜像、Git 或示例配置。
* 每项生产行为按 RED → GREEN → REFACTOR 执行。

---

### Task 1: 用预设 Provider 地址阻断 SSRF

**Files:**
- Create: src/finharness/server/provider_policy.py
- Modify: src/finharness/server/config_api.py
- Modify: src/finharness/config/settings.py
- Test: tests/server/test_config_api.py, tests/config/test_settings.py

**Interfaces:** 产生 validate_user_provider_url(base_url, kind, settings) -> str | None；创建、更新与 probe 三条入口共同调用。

- [ ] **Step 1: 写失败测试**

```python
def test_remote_config_rejects_an_unlisted_provider_origin(client):
    response = client.post("/v1/config", json=create_payload(base_url="https://169.254.169.254/v1"))
    assert response.status_code == 422
    assert response.json()["detail"]["errors"] == [
        {"field": "base_url", "message": "远程部署只允许使用运维预设的 Provider 地址"}
    ]
```

- [ ] **Step 2: 验证 RED**

Run: uv run pytest tests/server/test_config_api.py -k unlisted_provider_origin -q

Expected: FAIL；现有实现接受任意 HTTPS 地址。

- [ ] **Step 3: 最小实现**

fake 直接放行；本地保留绝对 HTTP(S) 格式校验；远程模式拒绝非 HTTPS，规范化末尾斜杠后，只允许 settings.providers 中的 base_url。ProviderSettings 在远程模式拒绝 HTTP；_field_errors 接收 settings，再由 create、update、probe 共用。

- [ ] **Step 4: 验证 GREEN**

Run: uv run pytest tests/server/test_config_api.py tests/config/test_settings.py -q

Expected: PASS；本地自定义测试端点仍可用。

- [ ] **Step 5: Commit**

git commit -m "fix: 限制远程 Provider 地址以阻断 SSRF"

### Task 2: 真实就绪检查与远程安全约束

**Files:**
- Modify: src/finharness/auth/store.py
- Modify: src/finharness/context/memory/store.py
- Modify: src/finharness/config/store.py
- Modify: src/finharness/server/api.py
- Modify: src/finharness/config/settings.py
- Test: tests/server/test_api.py, tests/config/test_settings.py

**Interfaces:** 每个 SQLite store 提供 ping() -> None；新增匿名 GET /v1/ready。

- [ ] **Step 1: 写失败测试**

```python
def test_ready_returns_ok_when_persistent_stores_are_usable(client):
    assert client.get("/v1/ready").json() == {"status": "ready"}

def test_ready_returns_503_when_memory_store_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(client.app.state.memory_store, "ping", lambda: (_ for _ in ()).throw(OSError("disk unavailable")))
    assert client.get("/v1/ready").status_code == 503
```

- [ ] **Step 2: 验证 RED**

Run: uv run pytest tests/server/test_api.py -k ready -q

Expected: FAIL；当前没有 ready 路由。

- [ ] **Step 3: 最小实现**

ping 使用自身连接执行 SELECT 1。ready 调用 UserStore、MemoryStore、ConfigStore 的 ping，并检查审计日志父目录可写；任一失败返回 503 和 {"status": "not_ready"}。health 保持 liveness。远程模式拒绝 secure_cookie=false。

- [ ] **Step 4: 验证 GREEN 并提交**

Run: uv run pytest tests/server/test_api.py tests/server/test_config_api.py tests/config/test_settings.py -q

Expected: PASS；提交信息：feat: 增加生产就绪检查。

### Task 3: Railway Docker 发布物

**Files:**
- Create: Dockerfile, .dockerignore, railway.toml, settings.railway.example.json
- Modify: README.md
- Test: tests/server/test_api.py

**Interfaces:** 产生 Docker 多阶段构建、/v1/ready healthcheck 与单 worker 启动命令；消费 PORT、RAILWAY_VOLUME_MOUNT_PATH 与 FINH_* Variables。

- [ ] **Step 1: 写失败测试并验证 RED**

新增远程模式未设安全 Cookie 时 create_production_app() 抛出 SettingsError 的测试。Run: uv run pytest tests/server/test_api.py -k secure_cookie -q。Expected: 在 Task 2 前 FAIL。

- [ ] **Step 2: 创建 Docker 和 Railway 配置**

Node 阶段执行 npm ci 和 npm run build；Python 阶段执行 uv sync --locked --no-dev --extra observability、复制 src 与 frontend/dist。Railway 启动固定一 worker，监听 0.0.0.0 和 PORT；railway.toml 的 healthcheckPath 是 /v1/ready。示例只列出远程模式、安全 Cookie、关闭公开注册、default 权限、以及 /data 下状态/缓存/输出/日志路径。Docker 使用非 root 用户；Railway 配置 RAILWAY_RUN_UID=0 以访问 root 挂载的 Volume。

- [ ] **Step 3: 验证构建并提交**

Run: npm --prefix frontend ci && npm --prefix frontend run build && docker build -t finharness:railway .

Expected: 镜像构建成功。提交信息：feat: 增加 Railway 单服务部署产物。

### Task 4: 一致性状态备份和 Railway 运维手册

**Files:**
- Create: scripts/backup_state.py
- Create: tests/scripts/test_backup_state.py
- Create: docs/deploy/railway.md

**Interfaces:** 产生 backup_state(state_dir: Path, destination: Path) -> Path 与 backup_state.py 的两个必填路径参数。

- [ ] **Step 1: 写失败测试并验证 RED**

```python
def test_backup_copies_sqlite_snapshot_and_master_key(tmp_path):
    state = make_state_with_users_db(tmp_path)
    snapshot = backup_state(state_dir=state, destination=tmp_path / "backups")
    assert (snapshot / "users.db").is_file()
    assert (snapshot / "secret.key").read_bytes() == (state / "secret.key").read_bytes()
```

Run: uv run pytest tests/scripts/test_backup_state.py -q

Expected: FAIL；模块不存在。

- [ ] **Step 2: 最小实现**

逐个数据库用 sqlite3.Connection.backup() 写入临时文件后原子重命名；复制 secret.key；生成含 UTC 时间和 SHA-256 的 manifest.json。拒绝缺少主密钥、目标不存在或目标位于 state 目录内；不上传备份、不读取 API Key 明文。

- [ ] **Step 3: 验证、写手册并提交**

Run: uv run pytest tests/scripts/test_backup_state.py -q && uv run python scripts/backup_state.py --help

Expected: PASS。手册列出 GitHub 自动部署、/data Volume、Variables、域名、/v1/ready、备份下载、停服恢复和单副本限制，并链接 Railway 官方文档。提交信息：feat: 增加状态备份与 Railway 运维手册。

### Task 5: CI 发布门禁与 Ruff 基线

**Files:**
- Modify: pyproject.toml, uv.lock, .github/workflows/eval.yml
- Modify: Ruff 报告的 src/ 与 tests/ 文件

- [ ] **Step 1: 先运行未来门禁，观察 RED**

Run: uv lock --check; uv run ruff check src tests; uv run ruff format --check src tests; uv run pytest -m "not smoke" -q; npm --prefix frontend ci; npm --prefix frontend test; npm --prefix frontend run build; npm --prefix frontend audit --omit=dev --audit-level=high; docker build -t finharness:ci .

Expected: Ruff 因现有问题和格式偏差失败。

- [ ] **Step 2: 清零 Ruff 并加入 CI**

先执行 ruff 自动修复和格式化；逐项消除余下 F811、F841、E402、E701/E702，不能用 noqa 屏蔽。将 ruff>=0.12 加入 dev extra 后更新锁文件。工作流新增 lint/format、前端 test/build/audit 和 Docker build。

- [ ] **Step 3: 验证 GREEN 并提交**

重新运行 Step 1 的命令，预期全为 0。全量 Python 用例若超出本地时限，必须取得 GitHub Actions 的 0 失败结果。提交信息：ci: 增加生产发布质量门禁。

### Task 6: 发布演练与文档同步

**Files:**
- Modify: README.md
- Modify: docs/modules/03.1-config.md
- Modify: docs/modules/03.12-server.md
- Modify: docs/modules/03.13-auth.md
- Modify: docs/modules/03.14-observability.md

- [ ] **Step 1: 更新说明**

明确本地和 Railway 远程模式；不得建议多 worker 或明文密钥；说明远程模式仅允许预设 Provider URL。

- [ ] **Step 2: 容器演练**

启动带本地 /data 挂载的镜像，传入远程模式、安全 Cookie 与 default 权限变量，请求 http://127.0.0.1:8001/v1/ready 并要求 200，随后停止容器。

- [ ] **Step 3: 最终验证与提交**

Run: git diff --check && uv run ruff check src tests && uv run ruff format --check src tests && npm --prefix frontend test && npm --prefix frontend run build

Expected: 全部通过。提交信息：docs: 更新 Railway 生产部署说明。

## Plan Self-Review

* Task 1 覆盖 SSRF 和 HTTPS；Task 2 覆盖 ready 与安全 Cookie；Task 3 覆盖 Docker、Volume、单 worker；Task 4 覆盖备份；Task 5 覆盖 CI 和质量门禁；Task 6 覆盖演练与文档。
* 没有 TBD、TODO 或“以后实现”占位符。
* Provider 策略只经 Settings 和 Config API 使用；备份工具不暴露 HTTP 路由；就绪检查只依赖既有 SQLite store。
