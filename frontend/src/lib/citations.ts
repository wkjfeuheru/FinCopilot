/** Turn `{cite:cit_000001}` markers into clickable, numbered references. */

const CITE_RE = /\{cite:(cit_\d+)\}/g;
// While a response streams, the tail may hold a marker that has not fully
// arrived yet (e.g. "{cite:cit_00"). Dropping it avoids a flash of raw syntax.
const PARTIAL_CITE_RE = /\{cite:[^}]*$/;

/**
 * Rewrite citations as markdown links to the matching source card
 * (`#cite-<cid>`). `order` maps a cid to its number in the source sidebar, so the
 * number shown here matches the card the link jumps to; unknown cids fall back to
 * first-seen numbering within this message.
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
