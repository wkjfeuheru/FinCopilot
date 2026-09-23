# 阿里云轻量应用服务器部署手册（香港节点，免备案）

> 适用架构：Docker Compose 跑应用容器 + Caddy 自动 TLS。单服务、单副本、
> 单 worker（与 Railway 方案同一约束：状态是进程内 SQLite，不可横向扩容）。
> 相关交付物：`Dockerfile`、`ops/docker-entrypoint.sh`、`deploy/docker-compose.yml`、
> `deploy/Caddyfile`、`deploy/.env.example`。

## 1. 购买服务器

1. 阿里云控制台 → 轻量应用服务器 → 创建实例：
   - 地域：**中国香港**（离内地近、免备案、LLM/数据源 API 延迟低）
   - 镜像：**Ubuntu 24.04**（或 22.04）
   - 套餐：**2 核 2G** 起步（应用加载 pandas/scipy，2G 是下限；预算允许可上 4G）
2. 创建后 → 防火墙 → 放行端口 **80、443**（22 默认已放行）
3. 记下公网 IP

## 2. 域名解析

在你的域名 DNS 处加一条 A 记录指向服务器公网 IP。
没有域名可在阿里云买一个（香港服务器无需备案，解析即用）。

## 3. 服务器装 Docker

SSH 登录后执行：

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # 重新登录生效
docker --version                 # 确认 27+ 
docker compose version           # 确认 v2 插件
```

## 4. 上传代码并配置密钥

```bash
# 方式 A：git 拉取（推荐）
sudo apt install -y git
git clone https://github.com/wkjfeuheru/FinCopilot.git
cd FinCopilot

# 方式 B：本机打包上传（GitHub 不可达时）
# 本机执行：git archive HEAD | ssh root@<IP> "mkdir -p ~/FinCopilot && tar -x -C ~/FinCopilot"
```

配置密钥（**只在服务器上做，绝不入库**）：

```bash
cp deploy/.env.example deploy/.env
nano deploy/.env        # 填 DOMAIN 和至少一个 Provider 密钥
chmod 600 deploy/.env
```

## 5. 构建并启动

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

首次构建约 5–10 分钟（93 个 Python 依赖 + 前端构建）。完成后验证：

```bash
docker compose -f deploy/docker-compose.yml ps        # 两个容器 Up (healthy)
curl -s http://127.0.0.1:8000/v1/ready                # {"status":"ready"}（经 docker exec 测试）
curl -s https://$DOMAIN/v1/ready                      # 公网 HTTPS 验证；Caddy 首次签证书需 ~1 分钟
```

浏览器打开 `https://<你的域名>`，应看到登录页。

## 6. 初始化管理员账号

注册默认关闭。临时打开：

```bash
sed -i 's/FINH_AUTH_ALLOW_REGISTER=false/FINH_AUTH_ALLOW_REGISTER=true/' deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

浏览器注册你的账号（用户名 2–32 字符，密码 ≥8 位）。**完成后立即改回**：

```bash
sed -i 's/FINH_AUTH_ALLOW_REGISTER=true/FINH_AUTH_ALLOW_REGISTER=false/' deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

## 7. 日常运维

```bash
cd ~/FinCopilot
git pull && docker compose -f deploy/docker-compose.yml up -d --build   # 发版
docker compose -f deploy/docker-compose.yml logs -f app                 # 看日志
docker compose -f deploy/docker-compose.yml restart app                 # 重启
```

数据都在命名卷 `deploy_app-data` 里（对应容器内 `/data`），重建容器不丢。
重新部署不丢账号、会话、加密配置——正是 `/data` 卷的用途。

**备份**（SQLite 在线快照工具是计划 Task 4 的交付物，落地前先用停机冷备）：

```bash
docker compose -f deploy/docker-compose.yml stop app
sudo tar -czf ~/backup-$(date +%F).tar.gz -C /var/lib/docker/volumes deploy_app-data/_data
docker compose -f deploy/docker-compose.yml start app
```

**恢复**：解包覆盖回同一卷路径后 `start app`。恢复到新服务器时，
先 `docker compose up -d caddy` 之外把数据卷放回原位再启动 app。

## 安全要点（与 Railway 方案同一套约束，已在镜像内强制）

- 所有非 fake Provider 必须 HTTPS；用户配置页只能选 `settings.providers`
  里预设的地址（SSRF 防护），自有密钥 Fernet 加密存储
- `secure_cookie`、`permission.default_mode=default` 由镜像内配置强制，
  无需手动设置
- 应用只监听容器网络（`expose` 而非 `ports`），公网入口只有 Caddy 的 80/443
- `.env` 含全部密钥：权限 600、绝不提交、换服务器时重新生成
