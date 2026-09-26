"""子代理协议：tools 与 coordinator 必须共同遵守的取值。

``spawn_agent`` 工具（tools 层）与 ``Coordinator``（coordinator 层）必须就"聚焦
名"与"单次派发上限"达成一致——工具负责声明并传参，协调器负责执行与校验。把这
几个取值放在 shared，使两侧的约定是一份声明，而不是一份散落的字面量：工具不必
反向依赖 coordinator 才能拿到它要传的 focus 名。

刻意只放**协议取值**，不放协调器的实现细节（各聚焦的轮次预算等留在
``coordinator/reviewer.py``）：tools 需要知道"派给谁、最多几个"，但不需要知道
"它跑多少轮"。

焦点分两类：``general`` 是模型可见的通用子代理（可取数，任务要求时也可联网），
``risk`` 与 ``reader`` 是内部角色（``risk`` 由 ``write_report`` 在进程内触发；
``reader`` 保留给内部只读材料任务，``summarize_document`` 不再内部派它），
不暴露给 ``spawn_agent``。
"""

from __future__ import annotations

# 风险复核者：一次独立阅读，能重新获取研报背后的数据（docs 03.10）。
RISK_FOCUS = "risk"
# 通用子代理：为上下文隔离而派生；任务点名了标的时可自行取数（docs 03.10）。
# 这是 `spawn_agent` 派出的唯一焦点，也是模型可见的唯一子代理角色。
GENERAL_FOCUS = "general"
# 内部 reader：只消化交给它的材料、不取数。保留给内部只读材料任务。
# ``summarize_document`` 不再内部派它——该工具只返回分片索引，由主 Agent
# 发现 spawn 后按片消化。**不暴露给模型**（模型侧只有 general）。
READER_FOCUS = "reader"

# 一次 spawn 调用最多可携带多少任务。每个任务都会变成一个并发的子代理并消耗
# 自己的 token，因此不设上限的列表会让单次调用耗尽整个会话预算。
MAX_SPAWN_TASKS = 8
