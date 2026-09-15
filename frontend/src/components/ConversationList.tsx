import { Button, Empty, List, Popconfirm, Typography } from "antd";
import type { ConversationSummary } from "../api/client";

type Props = {
  conversations: ConversationSummary[];
  activeId: string | null;
  loading: boolean;
  onSelect: (conversationId: string) => void;
  onNew: () => void;
  onRefresh: () => void;
  onDelete: (conversationId: string) => void;
};

function formatTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * 对话选择器。选中某一对话即恢复它：存储中保存着其记录，
 * 因此客户端会还原可见历史，并延续同一记忆作用域，
 * 而不是从头开始。
 */
export function ConversationList({
  conversations,
  activeId,
  loading,
  onSelect,
  onNew,
  onRefresh,
  onDelete,
}: Props) {
  return (
    <aside className="conversation-list">
      <div className="conversation-list-head">
        <Typography.Text strong>对话</Typography.Text>
        <Button size="small" type="primary" onClick={onNew}>
          新建对话
        </Button>
      </div>
      <Button size="small" type="link" disabled={loading} onClick={onRefresh}>
        {loading ? "加载中…" : "刷新"}
      </Button>
      {conversations.length === 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无历史对话" />
      ) : (
        <List
          size="small"
          dataSource={conversations}
          renderItem={(item) => (
            <List.Item
              className={item.conversation_id === activeId ? "conversation-item active" : "conversation-item"}
              onClick={() => onSelect(item.conversation_id)}
              actions={[
                // 删除具有破坏性且不可恢复，所以先询问确认。
                <Popconfirm
                  key="delete"
                  title="删除该对话？"
                  description="对话内容、引用与结论将一并删除，且不可恢复。"
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                  onConfirm={() => onDelete(item.conversation_id)}
                >
                  <Button
                    size="small"
                    type="link"
                    danger
                    onClick={(event) => event.stopPropagation()}
                  >
                    删除
                  </Button>
                </Popconfirm>,
              ]}
            >
              <div className="conversation-item-body">
                <span className="conversation-title">{item.title || "未命名对话"}</span>
                <span className="conversation-time">{formatTime(item.last_active_at)}</span>
              </div>
            </List.Item>
          )}
        />
      )}
    </aside>
  );
}
