# Theme tokens

## Compact token summary

- Background: `#edf1f5`; surface: `#fff`; ink: `#243649`; muted: `#637487`; border: `#d8e0e7`.
- Header: `#142638`; primary research accent: `#176b63`; soft accent: `#edf6f3`; success `#247552`; warning `#a97620`; error `#b64444`.
- Typography: `"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif`; data uses `"Cascadia Code", Consolas, ui-monospace, monospace`.
- Corners: 5–8px; shadows: restrained 2px soft shadows; desktop grid: `224px minmax(0, 1fr) 284px`.
- Breakpoints: `1180px` moves sources below; `700px` creates one-column mobile layout.

## Raw source reference

```css
:root { --ink:#243649; --muted:#637487; --line:#d8e0e7; --accent:#176b63; --accent-soft:#edf6f3; --surface:#fff; }
.app-header { color:#f4f7fa; background:#142638; border-bottom:3px solid #24796f; }
.workspace-body { display:grid; grid-template-columns:224px minmax(0,1fr) 284px; gap:20px; }
```
