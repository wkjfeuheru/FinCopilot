import { useEffect, useState } from "react";
import { Alert, Button, Form, Input, Modal, Select, Space, Switch, Tag, Typography } from "antd";
import {
  activateConfig,
  createConfig,
  deleteConfig,
  fetchPresets,
  probeConfig,
  updateConfig,
  type ConfigSnapshot,
  type Preset,
  type ProviderConfig,
} from "../api/config";

const KIND_LABELS: Record<string, string> = {
  openai_compat: "OpenAI 兼容",
  anthropic_compat: "Anthropic 兼容",
  fake: "离线 Fake",
};

type FormValues = {
  preset?: string;
  name: string;
  kind: string;
  base_url?: string;
  model: string;
  api_key?: string;
  activate: boolean;
};

type Props = {
  open: boolean;
  onClose: (savedAndActivated: boolean) => void;
  refreshConfig: () => Promise<ConfigSnapshot>;
};

export function SettingsModal({ open, onClose, refreshConfig }: Props) {
  const [form] = Form.useForm<FormValues>();
  const [presets, setPresets] = useState<Preset[]>([]);
  const [snapshot, setSnapshot] = useState<ConfigSnapshot>({ configured: false, active_id: null, configs: [] });
  const [editing, setEditing] = useState<ProviderConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const kind = Form.useWatch("kind", form);

  function resetForm() {
    setEditing(null);
    setProbe(null);
    form.resetFields();
    form.setFieldsValue({ kind: "openai_compat", activate: true });
  }

  async function loadTable() {
    setSnapshot(await refreshConfig());
  }

  useEffect(() => {
    if (!open) return;
    setError(null);
    setProbe(null);
    resetForm();
    void fetchPresets()
      .then((data) => setPresets(data.presets))
      .catch((exc: Error) => setError(exc.message));
    void loadTable();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  function applyPreset(name: string) {
    const preset = presets.find((item) => item.name === name);
    if (!preset) return;
    form.setFieldsValue({
      preset: name,
      name: preset.name,
      kind: preset.kind,
      base_url: preset.base_url ?? undefined,
      model: preset.name === "deepseek" ? "deepseek-chat" : form.getFieldValue("model"),
    });
  }

  function startEdit(record: ProviderConfig) {
    setEditing(record);
    setProbe(null);
    form.setFieldsValue({
      preset: undefined,
      name: record.name,
      kind: record.kind,
      base_url: record.base_url ?? undefined,
      model: record.model,
      api_key: undefined,
      activate: record.is_active,
    });
  }

  async function handleProbe() {
    const values = await form.validateFields(["kind", "model", "base_url", "api_key"]);
    setBusy(true);
    setProbe(null);
    try {
      const result = await probeConfig({
        kind: values.kind,
        base_url: values.base_url ?? null,
        model: values.model,
        env_key: presets.find((item) => item.name === values.preset)?.env_key ?? null,
        api_key: values.api_key || null,
        config_id: editing?.id,
      });
      setProbe(result.ok ? `连接成功 · ${result.latency_ms}ms` : `连接失败：${result.error}`);
    } catch (exc) {
      setProbe(`连接失败：${(exc as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleSave() {
    let values: FormValues;
    try {
      values = await form.validateFields();
    } catch {
      return;
    }
    setBusy(true);
    setError(null);
    const draft = {
      name: values.name.trim(),
      kind: values.kind,
      base_url: values.base_url ?? null,
      model: values.model.trim(),
      env_key: presets.find((item) => item.name === values.preset)?.env_key ?? null,
      api_key: values.api_key || null,
      activate: values.activate,
    };
    try {
      if (editing) await updateConfig(editing.id, draft);
      else await createConfig(draft);
      setBusy(false);
      onClose(values.activate);
      return;
    } catch (exc) {
      setError((exc as Error).message);
      setBusy(false);
    }
  }

  async function handleActivate(record: ProviderConfig) {
    setBusy(true);
    try {
      await activateConfig(record.id);
      await loadTable();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(record: ProviderConfig) {
    setBusy(true);
    try {
      await deleteConfig(record.id);
      if (editing?.id === record.id) resetForm();
      await loadTable();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title="模型供应商配置"
      open={open}
      onCancel={() => onClose(false)}
      width={960}
      footer={null}
      destroyOnHidden
    >
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        {error && <Alert type="error" message={error} showIcon closable onClose={() => setError(null)} />}
        {!snapshot.configured && (
          <Alert type="warning" message="尚未配置供应商，配置并启用后即可开始对话。" showIcon />
        )}

        <div className="settings-layout">
          <aside className="settings-config-list" aria-label="模型配置列表">
            <div className="settings-config-list-head">
              <div>
                <span className="settings-kicker">MODEL CONFIGURATIONS</span>
                <Typography.Title level={5}>已保存配置</Typography.Title>
              </div>
              <Button size="small" onClick={resetForm} disabled={busy}>新增</Button>
            </div>
            {snapshot.configs.length === 0 ? (
              <p className="settings-empty">尚无模型配置</p>
            ) : snapshot.configs.map((record) => (
              <article className={`settings-config-card ${editing?.id === record.id ? "active" : ""}`} key={record.id}>
                <button type="button" onClick={() => startEdit(record)}>
                  <strong>{record.name}</strong>
                  <span>{record.model}</span>
                  <small>{KIND_LABELS[record.kind] ?? record.kind}</small>
                </button>
                <div className="settings-config-tags">
                  {record.is_active && <Tag color="blue">使用中</Tag>}
                  <Tag color={record.has_key ? "green" : "default"}>{record.has_key ? "密钥已配置" : "缺少密钥"}</Tag>
                </div>
                <Space size="small" wrap>
                  {!record.is_active && <Button size="small" type="link" disabled={busy} onClick={() => void handleActivate(record)}>启用</Button>}
                  <Button size="small" type="link" disabled={busy} onClick={() => startEdit(record)}>编辑</Button>
                  <Button size="small" type="link" danger disabled={busy} onClick={() => void handleDelete(record)}>删除</Button>
                </Space>
              </article>
            ))}
          </aside>

          <div className="settings-form-panel">
            <div className="settings-form-head">
              <span className="settings-kicker">{editing ? "EDIT CONFIGURATION" : "NEW CONFIGURATION"}</span>
              <Typography.Title level={5}>{editing ? `编辑：${editing.name}` : "新增模型配置"}</Typography.Title>
              <p>密钥仅写入加密存储，页面不会回显。</p>
            </div>
            <Form<FormValues>
              form={form}
              layout="vertical"
              initialValues={{ kind: "openai_compat", activate: true }}
            >
          <Space align="start" wrap size="middle" style={{ width: "100%" }}>
            <Form.Item
              name="name"
              label="配置名称"
              rules={[{ required: true, message: "请填写配置名称" }]}
              style={{ minWidth: 200 }}
            >
              <Input placeholder="如 deepseek-prod" disabled={Boolean(editing)} />
            </Form.Item>
            <Form.Item name="preset" label="供应商预设" style={{ minWidth: 220 }}>
              <Select
                allowClear
                placeholder="选择预设自动填充"
                options={presets.map((preset) => ({
                  value: preset.name,
                  label: `${preset.name}（${KIND_LABELS[preset.kind] ?? preset.kind}${preset.has_env_key ? " · 环境变量可用" : ""}）`,
                }))}
                onChange={(value: string | undefined) => value && applyPreset(value)}
              />
            </Form.Item>
            <Form.Item name="kind" label="协议类型" rules={[{ required: true, message: "请选择协议类型" }]}>
              <Select
                style={{ minWidth: 180 }}
                options={Object.entries(KIND_LABELS).map(([value, label]) => ({ value, label }))}
              />
            </Form.Item>
            <Form.Item name="model" label="模型名称" rules={[{ required: true, message: "请填写模型名称" }]}>
              <Input placeholder="如 deepseek-chat" style={{ minWidth: 200 }} />
            </Form.Item>
          </Space>

          <Space align="start" wrap size="middle" style={{ width: "100%" }}>
            <Form.Item
              name="base_url"
              label="Base URL"
              rules={kind === "fake" ? [] : [{ required: true, message: "请填写 Base URL" }]}
              style={{ minWidth: 320 }}
            >
              <Input placeholder="https://api.deepseek.com/v1" disabled={kind === "fake"} />
            </Form.Item>
            <Form.Item
              name="api_key"
              label={editing ? "API Key（留空保持不变）" : "API Key"}
              style={{ minWidth: 280 }}
            >
              <Input.Password
                autoComplete="new-password"
                placeholder={editing?.has_key ? "已配置，留空保持不变" : "填写供应商密钥"}
              />
            </Form.Item>
            <Form.Item name="activate" label="保存后启用" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Space>

          {probe && (
            <Alert
              style={{ marginBottom: 12 }}
              type={probe.startsWith("连接成功") ? "success" : "error"}
              message={probe}
              showIcon
            />
          )}

          <Space>
            <Button type="primary" loading={busy} onClick={() => void handleSave()}>
              保存
            </Button>
            <Button disabled={busy} onClick={() => void handleProbe()}>
              测试连接
            </Button>
            {editing && (
              <Button disabled={busy} onClick={resetForm}>
                取消编辑
              </Button>
            )}
          </Space>
            </Form>
          </div>
        </div>
      </Space>
    </Modal>
  );
}
