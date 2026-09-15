/** 把 `{cite:cit_000001}` 标记转换为可点击的编号引用。 */

const CITE_RE = /\{cite:(cit_\d+)\}/g;
// 响应流式输出时，末尾可能含有尚未完整到达的标记
// （例如 "{cite:cit_00"）。丢弃它可避免闪现原始语法。
const PARTIAL_CITE_RE = /\{cite:[^}]*$/;

/**
 * 把 citation 改写为指向对应数据来源卡片的 markdown 链接
 * （`#cite-<cid>`）。`order` 把 cid 映射到它在数据来源侧栏中的编号，因此此处
 * 显示的编号与链接跳转到的卡片一致；未知 cid 则回退为
 * 本条消息内首次出现的编号。
 */
export function renderCitations(text: string, order: Map<string, number>): string {
  const local = new Map<string, number>();
  return text.replace(PARTIAL_CITE_RE, "").replace(CITE_RE, (_match, cid: string) => {
    let number = order.get(cid);
    if (number === undefined) {
      number = local.get(cid);
      if (number === undefined) {
        number = order.size + local.size + 1;
        local.set(cid, number);
      }
    }
    return `[[${number}]](#cite-${cid})`;
  });
}
