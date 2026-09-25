import { useState } from "react";
import { Alert, Button, Input, Modal, Space, Tag, Typography } from "antd";

import type { Interaction } from "../lib/interactionQueue";

export type { Interaction };

type Props = {
  interaction: Interaction | null;
  onRespond: (requestId: string, response: string) => void;
  busy: boolean;
  /** 队列中待处理的提示总数（含当前这条）。> 1 时提示用户还有后续。 */
  pendingCount?: number;
};

/**
 * 用于引擎暂停的两种情形的轮次中提示：写入工具请求
 * 确认（y/n），以及模型提出澄清问题。它们共用同一个
 * 通道，因此共用同一个对话框。
 */

/** 确认按钮的展示文案。
 *
 * ``y_remember`` 是网络访问确认的"本对话不再询问"应答（对话级）；
 * ``y_session`` 是写工具确认的"始终允许此类操作"应答（**会话级**：新会话即失效，
 * 因此文案必须点明"本会话内"，不能写成"不再询问"而让人误以为永久生效）。
 */
export function confirmLabel(option: string): string {
  if (option === "y") return "允许";
  if (option === "y_session") return "始终允许此类操作（本会话内）";
  if (option === "y_remember") return "允许并本对话不再询问";
  return "拒绝";
}

/** 模型有时会把"其他"直接列进 options；这类项应触发输入框而非当作普通答案提交。 */
const OTHER_LABELS = new Set(["其他", "其它", "other"]);
function isOtherOption(option: string): boolean {
  return OTHER_LABELS.has(option.trim().toLowerCase());
}

export function InteractionPrompt({ interaction, onRespond, busy, pendingCount = 1 }: Props) {
  const [freeText, setFreeText] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [otherOpen, setOtherOpen] = useState(false);

  if (!interaction) return null;
  const isConfirm = interaction.kind === "confirm";
  const options = interaction.options;
  const hasOptions = options.length > 0;
  const multi = interaction.multiSelect && !isConfirm;
  // 模型若已自行给出"其他"选项，复用它作为展开入口，不再重复追加。
  const otherOption = options.find(isOtherOption);
  const otherLabel = otherOption ?? "其他";
  const choices = options.filter((option) => !isOtherOption(option));
  const custom = freeText.trim();

  const submitCustom = () => {
    if (custom) onRespond(interaction.requestId, custom);
  };
  const submitMulti = () => {
    const parts = [...selected];
    if (otherOpen && custom) parts.push(custom);
    if (parts.length) onRespond(interaction.requestId, parts.join("；"));
  };
  const toggle = (option: string) =>
    setSelected((current) =>
      current.includes(option) ? current.filter((item) => item !== option) : [...current, option]
    );
  const canSubmitMulti = selected.length > 0 || (otherOpen && Boolean(custom));

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

        {!hasOptions && (
          <Space.Compact style={{ width: "100%" }}>
            <Input
              value={freeText}
              onChange={(event) => setFreeText(event.target.value)}
              placeholder="输入你的回答"
              onPressEnter={submitCustom}
              disabled={busy}
            />
            <Button type="primary" disabled={busy || !custom} onClick={submitCustom}>
              提交
            </Button>
          </Space.Compact>
        )}

        {hasOptions && isConfirm && (
          <Space wrap>
            {options.map((option) => (
              <Button
                key={option}
                type={option.startsWith("y") ? "primary" : "default"}
                danger={option === "n"}
                disabled={busy}
                onClick={() => onRespond(interaction.requestId, option)}
              >
                {confirmLabel(option)}
              </Button>
            ))}
          </Space>
        )}

        {hasOptions && !isConfirm && multi && (
          <>
            <Space wrap>
              {choices.map((option) => (
                <Button
                  key={option}
                  type={selected.includes(option) ? "primary" : "default"}
                  disabled={busy}
                  onClick={() => toggle(option)}
                >
                  {option}
                </Button>
              ))}
              <Button
                type={otherOpen ? "primary" : "default"}
                disabled={busy}
                onClick={() => setOtherOpen((open) => !open)}
              >
                {otherLabel}
              </Button>
            </Space>
            {otherOpen && (
              <Input
                value={freeText}
                onChange={(event) => setFreeText(event.target.value)}
                placeholder="输入你的回答"
                disabled={busy}
                autoFocus
              />
            )}
            <Button
              type="primary"
              block
              disabled={busy || !canSubmitMulti}
              onClick={submitMulti}
            >
              提交{selected.length > 0 ? `（已选 ${selected.length} 项）` : ""}
            </Button>
          </>
        )}

        {hasOptions && !isConfirm && !multi && otherOpen && (
          <Space.Compact style={{ width: "100%" }}>
            <Input
              value={freeText}
              onChange={(event) => setFreeText(event.target.value)}
              placeholder="输入你的回答"
              onPressEnter={submitCustom}
              disabled={busy}
              autoFocus
            />
            <Button type="primary" disabled={busy || !custom} onClick={submitCustom}>
              提交
            </Button>
            <Button
              disabled={busy}
              onClick={() => {
                setOtherOpen(false);
                setFreeText("");
              }}
            >
              返回选项
            </Button>
          </Space.Compact>
        )}

        {hasOptions && !isConfirm && !multi && !otherOpen && (
          <Space wrap>
            {choices.map((option) => (
              <Button
                key={option}
                disabled={busy}
                onClick={() => onRespond(interaction.requestId, option)}
              >
                {option}
              </Button>
            ))}
            <Button type="dashed" disabled={busy} onClick={() => setOtherOpen(true)}>
              {otherLabel}
            </Button>
          </Space>
        )}

        {isConfirm && (
          <Tag color="default">
            {pendingCount > 1 ? `还有 ${pendingCount - 1} 个待处理 · 超时未回答将视为拒绝并继续` : "超时未回答将视为拒绝并继续"}
          </Tag>
        )}
      </Space>
    </Modal>
  );
}
