import { useState } from "react";
import { Alert, Button, Form, Input, Tabs, Typography } from "antd";
import { login, register } from "../api/auth";

type Props = {
  /** 登录/注册成功后回传用户身份。 */
  onSuccess: (user: { id: string; username: string }) => void;
};

/** 未登录时的整页门：登录与注册共用一块表单。 */
export function LoginScreen({ onSuccess }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form] = Form.useForm<{ username: string; password: string }>();

  async function handleSubmit(values: { username: string; password: string }) {
    setBusy(true);
    setError(null);
    try {
      const session =
        mode === "login"
          ? await login(values.username.trim(), values.password)
          : await register(values.username.trim(), values.password);
      onSuccess(session.user);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "请求失败，请稍后重试");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <p className="eyebrow">FINANCIAL RESEARCH COPILOT</p>
        <h1>FinHarness</h1>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 18 }}>
          {mode === "login" ? "登录以继续你的研究对话" : "注册一个账号，开始你的研究对话"}
        </Typography.Paragraph>
        {error && (
          <Alert
            type="error"
            showIcon
            message={error}
            style={{ marginBottom: 14 }}
            closable
            onClose={() => setError(null)}
          />
        )}
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          requiredMark={false}
        >
          <Form.Item
            name="username"
            label="用户名"
            rules={[
              { required: true, message: "请输入用户名" },
              {
                min: 2,
                max: 32,
                message: "用户名长度须在 2-32 个字符之间",
              },
            ]}
          >
            <Input
              placeholder="字母、数字、下划线、连字符或中文"
              autoComplete="username"
              autoFocus
            />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: true, message: "请输入密码" },
              ...(mode === "register"
                ? [{ min: 8, message: "密码至少需要 8 个字符" } as const]
                : []),
            ]}
          >
            <Input.Password
              placeholder={mode === "register" ? "至少 8 个字符" : "密码"}
              autoComplete={mode === "register" ? "new-password" : "current-password"}
            />
          </Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            block
            loading={busy}
            style={{ marginTop: 4 }}
          >
            {mode === "login" ? "登录" : "注册并进入"}
          </Button>
        </Form>
        <Tabs
          activeKey={mode}
          centered
          style={{ marginTop: 8 }}
          items={[
            { key: "login", label: "已有账号" },
            { key: "register", label: "注册新账号" },
          ]}
          onChange={(key) => {
            setMode(key as "login" | "register");
            setError(null);
            form.resetFields();
          }}
        />
      </div>
    </div>
  );
}
