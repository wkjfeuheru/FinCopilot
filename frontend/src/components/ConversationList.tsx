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
 * The conversation picker. Selecting one resumes it: the store holds its
 * transcript, so the client restores the visible history and continues the same
 * memory scope rather than starting fresh.
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
                // Deleting is destructive and irreversible, so it asks first.
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
