import { authedFetch } from "./http";

/** 管理员页的 API 客户端（/v1/admin/*）。所有端点后端有 require_admin 兜底。 */

export type AdminUserRow = {
  id: string;
  username: string;
  role: "user" | "admin";
  created_at: string;
  conversations: number;
  last_active_at: string;
  turns: number;
  input_tokens: number;
  output_tokens: number;
  cache_hit_tokens: number;
  window_turns: number;
  window_input_tokens: number;
  window_output_tokens: number;
};

export type AdminUsersResponse = { users: AdminUserRow[]; window: string };

export type AdminUsageSummary = {
  window: string;
  total_users: number;
  turns: number;
  active_users: number;
  input_tokens: number;
  output_tokens: number;
  cache_hit_tokens: number;
};

export type AdminWindow = "all" | "24h" | "7d" | "30d";

/** 把失败响应转成可读错误：优先采用服务端的 detail（如"无管理员权限"）。 */
async function failure(response: Response, fallback: string): Promise<Error> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) return new Error(body.detail);
  } catch {
    // 非 JSON 错误体：退回状态码。
  }
  return new Error(`${fallback}：${response.status}`);
}

export async function fetchAdminStatus(): Promise<{ is_admin: boolean }> {
  const response = await authedFetch("/v1/admin/status");
  if (!response.ok) throw await failure(response, "加载管理状态失败");
  return (await response.json()) as { is_admin: boolean };
}

export async function fetchAdminUsers(window: AdminWindow = "all"): Promise<AdminUsersResponse> {
  const response = await authedFetch(`/v1/admin/users?window=${encodeURIComponent(window)}`);
  if (!response.ok) throw await failure(response, "加载用户总览失败");
  return (await response.json()) as AdminUsersResponse;
}

export async function fetchAdminSummary(window: AdminWindow = "all"): Promise<AdminUsageSummary> {
  const response = await authedFetch(`/v1/admin/usage/summary?window=${encodeURIComponent(window)}`);
  if (!response.ok) throw await failure(response, "加载用量汇总失败");
  return (await response.json()) as AdminUsageSummary;
}
