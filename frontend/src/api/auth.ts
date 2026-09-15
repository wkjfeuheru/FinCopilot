export type AuthUser = { id: string; username: string };

export type AuthSession = {
  token: string;
  user: AuthUser;
  expires_at: string;
};

async function post(path: string, body: unknown): Promise<AuthSession> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `请求失败：${response.status}`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      /* 保留基于状态码的提示信息 */
    }
    throw new Error(detail);
  }
  return (await response.json()) as AuthSession;
}

/** 注册并立即登录；服务端会同时下发会话 Cookie。 */
export function register(username: string, password: string): Promise<AuthSession> {
  return post("/v1/auth/register", { username, password });
}

/** 登录；服务端会同时下发会话 Cookie。 */
export function login(username: string, password: string): Promise<AuthSession> {
  return post("/v1/auth/login", { username, password });
}

/** 登出并撤销会话令牌。 */
export async function logout(): Promise<void> {
  await fetch("/v1/auth/logout", { method: "POST" });
}

/** 当前登录用户；401 表示未登录。 */
export async function fetchMe(): Promise<AuthUser | null> {
  try {
    const response = await fetch("/v1/auth/me");
    if (!response.ok) return null;
    const body = (await response.json()) as { user: AuthUser };
    return body.user;
  } catch {
    return null;
  }
}
