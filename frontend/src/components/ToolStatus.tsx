export function ToolStatus({ status }: { status: string | null }) {
  if (!status) return null;
  return <div className="tool-status">{status}</div>;
}
