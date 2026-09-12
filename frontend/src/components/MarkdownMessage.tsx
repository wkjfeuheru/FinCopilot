import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { artifactUrl } from "../api/client";
import { renderCitations } from "../lib/citations";

/** Sources a chart/report path produced on the server, not a remote URL. */
function isLocalFile(src: string): boolean {
  return !/^(https?:|data:|blob:|javascript:|#)/i.test(src);
}

function scrollToSource(cid: string) {
  document.getElementById(`cite-${cid}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
}

/** Renders an assistant message as markdown, with citations linked to sources. */
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
        // React-markdown sanitises unknown URL schemes by default, which would
        // strip Windows paths like C:\...; trust is restored here and the raw
        // link/image components below decide what is actually emitted.
        urlTransform={(url) => url}
        components={{
          img({ src, alt }) {
            const value = typeof src === "string" ? src : "";
            if (!value) return null;
            const resolved = isLocalFile(value) ? artifactUrl(value) : value;
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
            return (
              <div className="markdown-table-wrap">
                <table>{children}</table>
              </div>
            );
          },
        }}
      >
        {renderCitations(text, citationOrder)}
      </ReactMarkdown>
    </div>
  );
}
