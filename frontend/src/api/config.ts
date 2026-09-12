export type Preset = {
  name: string;
  kind: string;
  base_url: string | null;
  env_key: string | null;
  has_env_key: boolean;
};

export type ProviderConfig = {
  id: number;
  name: string;
  kind: string;
  base_url: string | null;
  model: string;
  env_key: string | null;
  is_active: boolean;
  has_key: boolean;
  created_at: string;
  updated_at: string;
};

export type ConfigSnapshot = {
  configured: boolean;
  active_id: number | null;
  configs: ProviderConfig[];
};

export type ProbeResult = {
  ok: boolean;
  latency_ms: number;
  model: string;
  error: string | null;
};

export type ConfigDraft = {
  name: string;
  kind: string;
  base_url: string | null;
  model: string;
  env_key: string | null;
  api_key: string | null;
  activate: boolean;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `请求失败：${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail?.errors) detail = body.detail.errors.map((e: { message: string }) => e.message).join("；");
    } catch {
      /* keep the status-based message */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function fetchConfig(): Promise<ConfigSnapshot> {
  return request<ConfigSnapshot>("/v1/config");
}

export function fetchPresets(): Promise<{ presets: Preset[] }> {
  return request<{ presets: Preset[] }>("/v1/config/presets");
}

export function createConfig(draft: ConfigDraft): Promise<{ config: ProviderConfig }> {
  return request<{ config: ProviderConfig }>("/v1/config", {
    method: "POST",
    body: JSON.stringify(draft),
  });
}

export function updateConfig(id: number, draft: ConfigDraft): Promise<{ config: ProviderConfig }> {
  return request<{ config: ProviderConfig }>(`/v1/config/${id}`, {
    method: "PUT",
    body: JSON.stringify(draft),
  });
}

export function activateConfig(id: number): Promise<{ config: ProviderConfig }> {
  return request<{ config: ProviderConfig }>(`/v1/config/${id}/activate`, { method: "POST" });
}

export function deleteConfig(id: number): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/v1/config/${id}`, { method: "DELETE" });
}

export function probeConfig(draft: Omit<ConfigDraft, "name" | "activate"> & { config_id?: number }): Promise<ProbeResult> {
  return request<ProbeResult>("/v1/config/probe", {
    method: "POST",
    body: JSON.stringify(draft),
  });
}
