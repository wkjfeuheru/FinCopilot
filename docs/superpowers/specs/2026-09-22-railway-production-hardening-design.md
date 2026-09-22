# Railway 生产化加固设计

## 目标

将 FinHarness 以 Railway 的单服务、单副本形态安全上线，并消除已发现的服务端请求伪造入口；发布产物、运行配置、健康检查、持久化与 CI 必须可重复验证。

## 范围

本次覆盖：Provider 配置的出站地址限制、远程模式配置约束、Railway Docker 发布产物、就绪检查、持久卷备份工具与说明、CI 发布门禁。

本次不覆盖：多副本/多区域部署、Redis 会话共享、PostgreSQL 迁移、用户角色系统、外部对象存储自动备份。这些能力需要单独设计，不能与单副本上线混合交付。

## 架构

Railway 运行一个 Docker 服务。构建阶段先使用 npm 锁文件生成 `frontend/dist`，运行阶段用锁定的 Python 依赖启动 FastAPI；Railway 终止 TLS，应用只监听 `0.0.0.0:$PORT`，且启动命令固定 `--workers 1`。前端静态文件和 `/v1` API 同源。

Railway Volume 挂载到 `/data`。状态库、Fernet 主密钥、每用户缓存与产物都由环境变量定位到该卷；服务重启或重新部署不会创建另一套账号、会话和密钥。应用仍保持单进程，不能横向扩容。

## Provider 地址策略

公网模式下，普通用户可以保存自己的 API Key 和选择模型，但 Provider `base_url` 必须精确匹配运维在 `settings.providers` 中声明的预设地址。后端在创建、更新和探测前统一执行该策略；`fake` 继续例外。这样不会依赖不可靠的“先 DNS 检查再连接”式 SSRF 防护，也不会允许用户借 `/probe` 访问内网地址。

远程模式同时要求所有非 fake Provider 使用 HTTPS、`auth.secure_cookie=true` 与 `permission.default_mode=default`。本地模式保留 HTTP 自定义 Provider 的开发便利性。

## 健康与配置

`/v1/health` 保持轻量 liveness；新增 `/v1/ready`，只在应用已经建立用户存储、记忆存储、配置密钥和审计日志目录均可用时返回 200。Railway 用 `/v1/ready` 作为部署健康检查。

生产设置由独立的 Railway 示例文件记录，密钥只通过 Railway Variables 注入。示例默认启用安全 Cookie、关闭公开注册、限制单租户并发流，并关闭 Prompt/响应捕获。是否开放注册由运维在上线时明确决定。

## 交付文件

* `Dockerfile`：多阶段构建，锁定 Node/Python 依赖，非 root 运行。
* `railway.toml`：Docker 构建、单 worker 启动命令和 `/v1/ready` 健康检查。
* `settings.railway.example.json`：不含密钥的 Railway 运行配置。
* `scripts/backup_state.py`：以 SQLite online backup API 创建一致性快照，并复制 `secret.key`；输出目录必须在 Railway Volume 内或由运维导出。
* `docs/deploy/railway.md`：Railway 账号、GitHub 连接、Volume、Variables、域名、备份和回滚步骤。

## CI 门禁

Pull request 与 main push 都必须执行：`uv lock --check`、Ruff lint/format、Python 离线测试、前端 `npm ci`/测试/构建、生产依赖审计、Docker 镜像构建。真实模型评估仍保留为定时和手动任务，避免每次推送产生费用。

## 测试

使用测试先行：先证明远程模式拒绝非预设与 HTTP 地址，再实现策略；覆盖创建、更新和 probe 三条入口。就绪检查覆盖可用与不可用依赖。备份脚本覆盖 SQLite 快照、主密钥复制和拒绝危险输出路径。Docker 与 Railway 配置由 CI 构建及文档中的启动/健康检查命令验证。

## 验收标准

1. 远程模式下，任意登录用户不能通过 Provider 配置或 probe 令服务端访问未声明的地址。
2. Railway 从干净构建可生成前端并启动单 worker 服务，`/v1/ready` 返回 200。
3. 重部署后，挂载 Volume 中的账号、会话、加密配置和报告仍存在。
4. CI 能阻止 Python、前端、依赖锁、格式检查或 Docker 构建失败的提交。
5. 备份产物可在隔离目录中包含一致的数据库快照和匹配的主密钥。
