# Agent Loop 显式 FSM 与事件驱动架构设计

日期：2026-09-25

## 目标

将 `AgentLoop` 当前由嵌套异步分支隐式表达的运行阶段改造成显式有限状态机，并用领域事件驱动状态转换。设计参考 Claude 的核心状态：

- `hydrate`
- `thinking`
- `tooluse`
- `awaitingconfirmation`
- `compact`
- `complete`
- `error`

每次状态转换都创建一个新的不可变状态对象。状态快照写入数据库，支持进程重启后恢复。现有 `AgentLoop.run()`、`AgentTurnOutcome`、SSE 过程事件及工具执行语义保持向后兼容。

## 非目标

- 不引入 Kafka、RabbitMQ 等外部消息代理。
- 不把 provider 的每个流式 chunk 持久化为状态。
- 不采用完整 Event Sourcing；领域事件驱动转换，但恢复读取状态快照。
- 不重写 provider、tool、compactor、memory、observer 的内部实现。
- 不改变用户停止不是错误、工具普通失败不终止运行等既有语义。

## 现状与约束

当前控制流为 `run -> _run_once -> _run_turns -> _execute_one`。目标状态分散在以下位置：

- hydrate：记忆加载、断点恢复、语义召回和方法论路由。
- compact：每轮模型请求前的上下文压缩。
- thinking：provider 流式请求。
- tooluse：并行工具执行。
- awaitingconfirmation：隐藏在 `PermissionGate` 或 `AskUserTool` 调用的交互回调中。
- complete/error：最终回答、失败、停止和轮次耗尽分支。

重构必须保持以下不变量：

1. assistant tool-use 消息必须有完整的 tool-result 配对。
2. 工具轮中流出的文本是草稿；进入工具执行前仍发送 `text_reset`。
3. compaction 只发生在轮次边界，不发生在 provider 请求中途。
4. 同一轮的 prompt 状态视图保持一致。
5. 用户停止不发送 `error`，已有发现会持久化，并保持可恢复。
6. 硬取消仍持久化用户消息、已完成结果和可恢复断点。
7. 工具调用继续并发执行，并保持结果按原 tool-use 顺序回填。
8. 审计、进展和输出通道失败不破坏主要执行；状态数据库写入失败除外。
9. 已完成 checkpoint 不得把旧计划泄漏到新请求。
10. 主循环、子 Agent、eval 和脚本继续共用相同的引擎接口。

## 方案选择

采用“不可变状态快照 + 纯 transition reducer + effect runner”。

未采用每状态一个类的 State Pattern，因为共享上下文、并发工具状态和序列化会产生较多样板。未采用完整 Event Sourcing，因为现有消息、trace、checkpoint 已有独立持久化语义，迁移为事件重放会扩大本次范围。

## 架构

### 模块

`finharness.engine.state`

- 定义 `AgentPhase`、`AgentState`、工具调用恢复记录、确认记录、终态记录和领域事件。
- 提供纯函数 `transition(state, event) -> AgentState`。
- 不依赖 provider、tool、数据库或输出通道。

`finharness.engine.machine`

- 提供串行化的 `dispatch(event)`。
- 校验 revision 和合法转换。
- 调用 reducer 生成新对象。
- 先持久化，再发送 `EngineEvent(kind="state", ...)`。

`finharness.engine.runner`

- 根据当前 phase 执行 hydrate、provider、tool、interaction、compaction 等副作用。
- 把副作用结果转换为领域事件并发布给 machine。
- 管理仅在当前进程内有意义的 Task、Future、流式 delta、计时器和 Span。

`finharness.context.memory.state_store`

- 持久化状态快照。
- 查询某个 run 或 conversation 的最新状态。
- 处理 schema version、用户隔离、对话删除和 retention。

`InteractivePort`

- 统一权限确认和 `ask_user`。
- 请求用户之前发布确认请求领域事件，使 FSM 先进入 `awaitingconfirmation`。
- 服务端使用 ConfirmBus adapter；eval 和测试使用各自 adapter。

`AgentLoop`

- 保留现有构造方式和 `run()` 接口。
- 作为兼容门面组装 machine/runner，并继续提供现有辅助能力。

### 运行数据流

```text
Effect 完成
  -> 发布 AgentEvent
  -> reducer 计算新 AgentState
  -> 数据库事务提交状态及关联消息
  -> emit("state", 公共状态视图)
  -> runner 调度下一状态的 Effect
```

领域事件通过进程内异步队列串行归约。并行工具可以并发完成，但每个完成事件依次进入 reducer，因此 revision 单调递增且状态更新不会相互覆盖。

## 状态模型

`AgentState` 是冻结的不可变对象，至少包含：

- `schema_version`
- `run_id`
- `conversation_id`
- `user_id`
- `revision`
- `phase`
- `turn`
- usage、tool call、retry、compaction 累计计数
- `calls`：call id、工具名、参数、权限级别、执行状态、已持久化结果
- `confirmation`：提示、选项、关联 call id 和确认状态
- `outcome`：成功状态、终止原因、是否可恢复
- `error`：结构化错误类别和可安全展示的消息
- `created_at`

以下内容不得进入状态对象：

- provider 流式 delta 或 chunk
- asyncio Task、Future、Event、Queue 或锁
- 输出 sink
- Span、活动计时器和回调
- provider、registry、tool、store 等运行时对象
- 当前进程生成的确认 `request_id`

消息、citation、summary、LTM 继续使用现有表和模型；状态快照只保存恢复执行所需的信息，不复制这些内容。

## 状态转换

合法主转换为：

```text
hydrate -> compact | thinking | tooluse | awaitingconfirmation
compact -> thinking | tooluse | awaitingconfirmation
  （后两者仅当快照存有 resume_phase ∈ {tooluse, awaitingconfirmation}；
   resume_phase 是恢复字段，不是第二个公开阶段）
thinking -> tooluse | complete | error
tooluse -> awaitingconfirmation | thinking | complete | error
awaitingconfirmation -> tooluse | complete | error
任何 resumable 快照 -> hydrate（仅由显式 ResumeRequested 触发）
```

补充规则：

- `complete` 和 `error` 是静止终态，不会自动调度副作用；只有带
  `resumable=true` 的快照可被外部 `ResumeRequested` 显式带回 `hydrate`。
- 新用户请求创建新 run，并从 `hydrate` 开始。
- 恢复请求沿用旧 `run_id`，先进入 `hydrate` 校验持久化状态，再回到应恢复的阶段。
- 有任一工具等待用户时，全局 phase 为 `awaitingconfirmation`；各工具的并发状态保存在 `calls`。
- 普通工具错误、权限拒绝和超时是工具结果，不进入 FSM `error`。
- provider 不可恢复错误、非法转换、损坏状态和必要状态持久化失败进入 `error`。
- compaction 失败按现有语义降级并继续 `thinking`。
- 用户停止进入 `complete`，但 `outcome` 为 `stopped` 且 `resumable=true`；不发送 `error`。
- 用户直接开始新请求时，旧非终态 run 转为 `complete/outcome="abandoned"`。

## 持久化

### 表结构

新增 append-only 的 `agent_state_snapshots` 表。每次合法转换追加一行，以 `(run_id, revision)` 唯一。至少为 `(conversation_id, created_at)` 和 `(run_id, revision)` 建索引。

每行包含身份字段、phase、revision、schema version、序列化状态载荷和时间戳。用户归属必须参与所有 conversation 查询，避免跨租户读取恢复状态。

恢复读取最新的 `resumable=true` 快照；这既包括进程崩溃留下的非终态，
也包括用户停止产生的 `complete/outcome="stopped"`。历史快照按现有
conversation retention 规则清理；删除 conversation 时级联删除。快照是
恢复来源，但不是 UI 的无限期审计日志；长期观测仍由 TraceStore 承担。

### 原子提交

- `thinking -> tooluse`：assistant tool-use 消息与新快照在同一事务提交。
- 单个工具完成：结果先写入新快照。
- 工具批次全部完成：按原调用顺序生成一个 tool-result 消息，并与 `tooluse -> thinking` 在同一事务提交。
- 用户确认决定：先持久化，再开始对应工具副作用。
- 终态：最终 assistant 消息、终态快照和兼容 checkpoint 在同一事务边界收敛。

状态写入失败时，不执行下一项副作用。若连 error 快照也无法持久化，异常交由服务端现有终止帧兜底处理。

## 进程重启恢复

恢复必须由客户端显式请求，不根据“继续”等自然语言推断。

`ChatRequest` 增加 `resume: bool = false`：

- `resume=true`：对最新可恢复快照发布 `ResumeRequested`，沿用原 run id 和递增 revision 进入 hydrate。
- `resume=false` 且存在旧非终态 run：先将旧 run 标为 abandoned，再创建新 run。

对话历史接口返回最新状态的安全公共视图和 `resumable` 标记，使客户端能展示“继续研究”入口。

各 phase 的恢复策略：

- hydrate：重新执行幂等的加载和一致性校验。
- thinking：丢弃未完成的瞬时文本，重新发起该模型轮次。
- compact：重新执行 compaction；compactor 必须继续保持幂等覆盖语义。
- tooluse：不执行已有持久化结果的调用；只读 pending 调用按原 call id 重试。
- awaitingconfirmation：使用持久化的提示、选项和工具信息重新签发确认，生成新的临时 request id。
- complete/error：不继续执行。

若进程在写工具开始后、结果持久化前崩溃，其实际效果未知：

- 只读工具可以自动重试。
- 写工具不得自动重放。
- 写工具恢复为 awaitingconfirmation，并明确提示上次执行结果未知；用户再次确认后才重试。
- 已有持久化结果的调用绝不重复执行。

该方案提供安全的至少一次恢复语义，但不宣称外部副作用具有 exactly-once 保证。

## 事件契约

新增兼容事件：

```text
EngineEvent(kind="state", data=<public state view>)
```

每次状态转换完成数据库提交后发送一次。公共视图包含 run id、revision、phase、turn、调用摘要、确认摘要和终态摘要，但不包含完整工具参数、内部错误细节或其他敏感字段。

现有事件继续保留，包括：

- `text_delta` / SSE `delta`
- `text_reset`
- `tool_status`
- `tool_progress`
- `tool_activated`
- `interactive_request`
- `interaction_resolved`
- `context_compacted`
- `context_routed`
- `plan_progress`
- `loop_guard`
- `answer`
- `error`
- `done`

顺序要求：

- 状态提交和 `state(awaitingconfirmation)` 先于 `interactive_request`。
- `interaction_resolved` 先于或紧邻恢复到 `tooluse` 的 state 事件，但确认决定必须先持久化。
- `state(complete|error)` 在终止 `done` 之前发出。
- 输出失败不回滚已提交状态；刷新后以数据库状态为准。

QueueSink、TraceStore、历史重放和前端 reducer 增加对 `state` 的支持。前端以 `state` 作为主要阶段来源，旧过程事件继续用于详细进度和旧服务端兼容。

## 错误处理

- reducer 遇到非法转换时产生明确的 invariant failure，不静默修正。
- 未知 schema version 进入安全错误状态，不使用猜测字段恢复。
- 状态载荷损坏时保留原记录供诊断，并阻止副作用重放。
- provider 在首 chunk 前的重试语义保持不变；流式中途失败不自动重试。
- 工具、hook、audit 和 progress 的既有错误等级保持不变。
- 硬取消先持久化当前可恢复状态和完整工具配对信息，再向上传播取消。
- 服务端仍保证无论引擎如何退出都发送或合成终止帧。

## 兼容与迁移

- `AgentLoop.run(user_msg)` 保持可用；增加可选 resume 参数或等价命令对象时必须有默认值。
- `AgentTurnOutcome` 保持现有字段与语义。
- 现有 checkpoint 可兼容读取。FSM snapshot 上线后，checkpoint 作为旧数据和旧调用方兼容层，逐步由状态快照取代。
- `EngineEvent.kind` 仍允许字符串；本次只增加 `state`，不强制一次性迁移所有事件为枚举。
- eval、脚本和子 Agent 未注入持久化 store 时使用内存 StateStore adapter，行为与当前无数据库模式一致。
- 当前工作区中与本设计无关的修改不纳入实现范围。

## 测试策略

### 纯状态机

- 所有合法转换。
- 所有非法转换。
- 每次转换返回新对象，旧对象保持不变。
- revision 单调递增。
- 序列化、反序列化和 schema version。
- 公共视图不泄露工具参数。

### 引擎集成

- hydrate、compact、thinking、tooluse、awaitingconfirmation、complete、error 的事件序列。
- 多工具并行完成时 reducer 串行归约，tool-result 顺序仍与 tool-use 一致。
- 工具草稿 reset、assistant/tool-result 配对和 prompt 状态一致性。
- 用户停止无 error、部分成果保留、终态可恢复。
- provider 首 chunk 前重试和流中失败。
- compaction 降级不阻断运行。

### 恢复

- 每个非终态 phase 的数据库往返和重启恢复。
- awaitingconfirmation 重签 request id。
- 只读 pending 工具自动重试。
- 写工具未知结果要求再次确认。
- 已完成工具不重复执行。
- resume=false 将旧 run 标为 abandoned。
- resume=true 沿用 run id 和 revision。
- conversation 删除、用户隔离和 retention。

### 消费者

- SSE 保留现有事件并新增 state。
- QueueSink replay 包含 state。
- TraceStore 记录 state。
- 前端 live 与历史重放得到一致 phase。
- 旧客户端忽略 state 后仍能正常完成对话。
- eval 和子 Agent 在内存 adapter 下正常运行。

### 回归套件

重点运行 engine loop、checkpoint、user stop、compaction、governance、interactive、SSE、trace store、frontend trace 和 eval 测试，并执行项目完整测试套件。

## 验收标准

1. 七个核心 phase 在代码和运行事件中均显式可见。
2. 每次合法转换产生新的不可变对象和单调 revision。
3. 每次转换在调度下一副作用前写入数据库。
4. 服务重启后可显式恢复非终态运行。
5. awaitingconfirmation 恢复后重新提示，不复用旧 request id。
6. 未知写副作用不会被静默自动重放。
7. 现有事件、返回值和主要行为保持兼容。
8. 现有不变量和新增 FSM/恢复测试全部通过。
