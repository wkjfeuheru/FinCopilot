import { useState } from "react";
import { Alert, Button, Input, Modal, Space, Tag, Typography } from "antd";

export type Interaction = {
  requestId: string;
  kind: string;
  prompt: string;
  options: string[];
};

type Props = {
  interaction: Interaction | null;
  onRespond: (requestId: string, response: string) => void;
  busy: boolean;
};

/**
 * Mid-turn prompt for both cases the engine pauses on: a write tool asking for
 * confirmation (y/n) and the model asking a clarifying question. They share one
 * channel, so they share one dialog.
 */
export function InteractionPrompt({ interaction, onRespond, busy }: Props) {
  const [freeText, setFreeText] = useState("");

  if (!interaction) return null;
  const isConfirm = interaction.kind === "confirm";
  const hasOptions = interaction.options.length > 0;

  return (
    <Modal
      open
      title={isConfirm ? "需要确认" : "模型提问"}
      footer={null}
      closable={false}
      maskClosable={false}
      width={520}
      destroyOnHidden
    >
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Alert
          type={isConfirm ? "warning" : "info"}
          showIcon
          message={isConfirm ? "该操作需要你批准后才会执行" : "回答后模型将继续本次任务"}
        />
        <Typography.Paragraph style={{ margin: 0, whiteSpace: "pre-wrap" }}>
          {interaction.prompt}
        </Typography.Paragraph>

        {hasOptions ? (
          <Space wrap>
            {interaction.options.map((option) => (
              <Button
                key={option}
                type={isConfirm && option === "y" ? "primary" : "default"}
                danger={isConfirm && option === "n"}
                disabled={busy}
                onClick={() => onRespond(interaction.requestId, option)}
              >
                {isConfirm ? (option === "y" ? "允许" : "拒绝") : option}
              </Button>
            ))}
          </Space>
        ) : (
          <Space.Compact style={{ width: "100%" }}>
            <Input
              value={freeText}
              onChange={(event) => setFreeText(event.target.value)}
              placeholder="输入你的回答"
              onPressEnter={() => {
                if (freeText.trim()) onRespond(interaction.requestId, freeText.trim());
              }}
              disabled={busy}
            />
            <Button
              type="primary"
              disabled={busy || !freeText.trim()}
              onClick={() => onRespond(interaction.requestId, freeText.trim())}
            >
              提交
            </Button>
          </Space.Compact>
        )}

        {isConfirm && <Tag color="default">超时未回答将视为拒绝并继续</Tag>}
      </Space>
    </Modal>
  );
}
