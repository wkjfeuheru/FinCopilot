/** 引擎暂停、等待用户作答的提示队列。
 *
 * 同一轮里并发的多个工具确认或提问会先后到达。此前组件用单值 state 保存
 * 当前提示，后到的会覆盖先到的，那条提示的 request_id 随之消失、再也无法
 * 被应答，只能等满服务端 TTL 被判拒绝，并把整轮拖住。这里改成队列：全部
 * 保留，一次只呈现队首。
 */

export type Interaction = {
  requestId: string;
  kind: string;
  prompt: string;
  options: string[];
  /** 多选提问：选项可勾选多项，提交时以「；」连接。 */
  multiSelect: boolean;
};

/** 追加一条提示；同 request_id 重复到达时忽略（合并请求会多次下发同一 id）。 */
export function enqueue(queue: Interaction[], item: Interaction): Interaction[] {
  if (!item.requestId) return queue;
  if (queue.some((existing) => existing.requestId === item.requestId)) return queue;
  return [...queue, item];
}

/** 移除一条已落定的提示；requestId 为空（如整轮出错）时清空整个队列。 */
export function resolve(queue: Interaction[], requestId: string | null): Interaction[] {
  if (!requestId) return [];
  return queue.filter((existing) => existing.requestId !== requestId);
}
