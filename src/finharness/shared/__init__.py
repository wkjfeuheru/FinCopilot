"""跨 tools 与 engine 共享的纯函数与声明：无状态、无 I/O。

这个包是为了消除两处真实的逆向依赖而存在的——它们都是"被依赖方比依赖方还靠上"
造成的：

* ``tools → coordinator``：``tools/fin/writer.py`` 用研报复核的渲染与编排
  （``review``），``tools/meta/summarize.py`` 用分片归并（``summarize``）与数字
  缺漏检测（``numbers``）。这三者是纯函数或纯编排，与多代理无关。
* ``engine → tools``：``engine/loop.py`` 需要工具声明层（``declaration``）、
  按能力的意图推断（``capabilities``）、结果预算解析（``budget``）与围栏中和
  （``fencing``）。这些既要被 engine 依赖、又要被 tools 依赖，放在 tools 里
  就形成 engine ↔ tools 的环。

因此本包位于 tools 与 engine 之下，二者都只向下依赖它：

    shared/
      ├─ declaration.py   @tool / @param 与 ToolSpec、DECLARED_TOOLS
      ├─ capabilities.py  从文本推断能力意图（RESEARCH_CAPABILITIES 等）
      ├─ budget.py        工具结果 token 预算的单点解析
      ├─ fencing.py       第三方文本围栏与中和
      ├─ numbers.py       数字缺漏检测（原 coordinator/numbers.py）
      ├─ review.py        研报复核的渲染与编排（原 coordinator/review.py）
      └─ summarize.py     长文档结构感知切分（分片 id / 骨架；原 coordinator/summarize.py）

依赖方向：``shared`` 只允许依赖 ``utils`` / ``config`` / ``types``；不得依赖
``tools`` / ``engine`` / ``coordinator`` / ``data`` / ``context`` / ``server``。
该规则由 ``.importlinter`` 的契约固化。

``coordinator/`` 保留为多代理编排（``reviewer.py``：派生隔离子代理），它是
tools 与 engine 之上的消费者，不再被二者反向依赖。
"""
