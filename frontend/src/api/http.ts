/** 认证失败时派发的全局事件；App 监听它回到登录页。 */
export const AUTH_EXPIRED_EVENT = "finharness:auth-expired";

/**
 * 统一的 fetch 包装。
 *
 * 同源部署下浏览器自动携带会话 Cookie，因此这里不需要任何令牌管理；
 * 唯一的额外职责是把 401 广播出去，让整棵应用树回到登录页，
 * 而不是让每个调用点各自吞掉这个状态。
 */
export async function authedFetch(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(path, init);
  if (response.status === 401) {
    window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT));
  }
  return response;
}
