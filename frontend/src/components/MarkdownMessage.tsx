import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { artifactUrl } from "../api/client";
import { renderCitations } from "../lib/citations";

/** 图表/研报路径由服务端产出，并非远程 URL。 */
function isLocalFile(src: string): boolean {
  return !/^(https?:|data:|blob:|javascript:|#)/i.test(src);
}

// react-markdown 在 URL 到达这里之前会做百分号编码（normalizeUri：
// `F:\a b.png` 会以 `F%3A%5Ca%20b.png` 到达）。服务端需要原始路径才能
// 解析产出物，因此这里解码回去；对于并非合法转义的字面 `%`，
// 通过兜底分支避免 decodeURIComponent 出现未定义行为。
function decodeLocalPath(src: string): string {
  try {
    return decodeURIComponent(src);
  } catch {
    return src;
  }
}

// 链接目标不能包含未转义的空格，因此像
// `F:\python project\...png` 这样的路径会让整个 `![alt](path)` 渲染为
// 字面文本而非图片。尖括号是规范为此类目标提供的转义方式；
// 将本地文件路径包裹起来，图表才能显示为图片。
const LOCAL_DESTINATION_RE = /(!?\[[^\]]*\]\()([^()<>\n]+)(\))/g;

function wrapLocalDestinations(text: string): string {
  return text.replace(
    LOCAL_DESTINATION_RE,
    (match: string, head: string, rawDest: string, tail: string) => {
      const dest = rawDest.trim();
      if (!dest || !/\s/.test(dest)) return match;
      if (/^(https?:|data:|blob:|mailto:|tel:|#)/i.test(dest)) return match;
      // 只有路径才由我们重新解释；普通链接保持不动。
      if (!dest.includes("\\") && !dest.includes("/")) return match;
      return `${head}<${dest}>${tail}`;
    },
  );
}

function scrollToSource(cid: string) {
  document.getElementById(`cite-${cid}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
}

/** 将助手消息渲染为 markdown，并把 citations 链接到数据来源。 */
export function MarkdownMessage({
  text,
  citationOrder,
}: {
  text: string;
  citationOrder: Map<string, number>;
}) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        // react-markdown 默认会净化未知的 URL scheme，这会
        // 剥离像 C:\... 这样的 Windows 路径；这里恢复信任，由下方的
        // 原生链接/图片组件决定实际输出什么。
        urlTransform={(url) => url}
        components={{
          img({ src, alt }) {
            const value = typeof src === "string" ? src : "";
            if (!value) return null;
            const resolved = isLocalFile(value)
              ? artifactUrl(decodeLocalPath(value))
              : value;
            return <img src={resolved} alt={alt ?? ""} loading="lazy" />;
          },
          a({ href, children }) {
            const target = href ?? "";
            if (target.startsWith("#cite-")) {
              return (
                <a
                  className="cite-ref"
                  href={target}
                  onClick={(event) => {
                    event.preventDefault();
                    scrollToSource(target.slice("#cite-".length));
                  }}
                >
                  {children}
                </a>
              );
            }
            if (target.startsWith("javascript:")) return <span>{children}</span>;
            return (
              <a href={target} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
          table({ children }) {
            // 表头随纵向滚动固定：财务对照表动辄几十行，滚到中部时
            // 列名还在，数字才有可读性。表头吸附要求容器是独立滚动
            // 盒（sticky 不越过 overflow 祖先），因此这里限高内滚。
            return (
              <div className="markdown-table-wrap">
                <div className="markdown-table-scroll">
                  <table>{children}</table>
                </div>
              </div>
            );
          },
        }}
      >
        {wrapLocalDestinations(renderCitations(text, citationOrder))}
      </ReactMarkdown>
    </div>
  );
}
