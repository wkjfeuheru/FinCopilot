import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { artifactUrl } from "../api/client";
import { renderCitations } from "../lib/citations";

/** Sources a chart/report path produced on the server, not a remote URL. */
function isLocalFile(src: string): boolean {
  return !/^(https?:|data:|blob:|javascript:|#)/i.test(src);
}

// react-markdown percent-encodes URLs before they reach here (normalizeUri:
// `F:\a b.png` arrives as `F%3A%5Ca%20b.png`). The server needs the raw path to
// resolve the artefact, so decode it back; a literal `%` that is not a valid
// escape leaves decodeURIComponent undefined-behaviour-safe via the fallback.
function decodeLocalPath(src: string): string {
  try {
    return decodeURIComponent(src);
  } catch {
    return src;
  }
}

// A link destination may not contain an unescaped space, so a path like
// `F:\python project\...png` makes the whole `![alt](path)` render as literal
// text instead of an image. Angle brackets are the spec's escape hatch for such
// destinations; wrap the local file paths so charts show up as images.
const LOCAL_DESTINATION_RE = /(!?\[[^\]]*\]\()([^()<>\n]+)(\))/g;

function wrapLocalDestinations(text: string): string {
  return text.replace(
    LOCAL_DESTINATION_RE,
    (match: string, head: string, rawDest: string, tail: string) => {
      const dest = rawDest.trim();
      if (!dest || !/\s/.test(dest)) return match;
      if (/^(https?:|data:|blob:|mailto:|tel:|#)/i.test(dest)) return match;
      // Only paths are ours to reinterpret; prose links stay untouched.
      if (!dest.includes("\\") && !dest.includes("/")) return match;
      return `${head}<${dest}>${tail}`;
    },
  );
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
            return (
              <div className="markdown-table-wrap">
                <table>{children}</table>
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
