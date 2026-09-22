type ResearchWelcomeProps = {
  onUsePrompt: (prompt: string) => void;
};

const SCENARIOS = [
  {
    code: "EQUITY",
    title: "个股研究",
    summary: "财务体检、盈利拆解、估值与同业对比",
    prompt: "分析贵州茅台的盈利质量与当前估值",
  },
  {
    code: "INDUSTRY",
    title: "行业研究",
    summary: "产业链、竞争格局、景气度与行业空间",
    prompt: "分析光伏行业当前景气度与竞争格局",
  },
  {
    code: "MACRO",
    title: "宏观研究",
    summary: "经济周期、政策解读与资产传导",
    prompt: "近期社融结构变化如何影响风险偏好？",
  },
  {
    code: "FACTOR",
    title: "量化因子",
    summary: "策略回测、样本内外验证与过拟合检查",
    prompt: "验证价值因子在 A 股近三年的稳定性",
  },
];

export function ResearchWelcome({ onUsePrompt }: ResearchWelcomeProps) {
  return (
    <section className="research-welcome" aria-labelledby="research-welcome-title">
      <div className="research-mandate">
        <div className="research-mandate-head">
          <div>
            <span className="research-kicker">RESEARCH MANDATE</span>
            <h2 id="research-welcome-title">建立一份可复核的研究委托</h2>
          </div>
          <span className="evidence-badge">证据优先</span>
        </div>
        <p>
          从明确问题开始，研究过程将记录计划、数据证据、结论依据与风险复核。输出仅用于研究参考，不构成投资建议。
        </p>
        <ol className="research-chain" aria-label="研究链路">
          {[
            ["问题", "当前"],
            ["计划", ""],
            ["取证", ""],
            ["结论", ""],
            ["复核", ""],
          ].map(([label, state], index) => (
            <li className={state ? "active" : ""} key={label}>
              <span>{label}</span>
              {index < 4 && <i aria-hidden="true" />}
            </li>
          ))}
        </ol>
      </div>

      <div className="scenario-heading">
        <div>
          <span className="research-kicker">RESEARCH ENTRY</span>
          <h2>选择研究场景</h2>
        </div>
        <span>点击示例直接开始研究</span>
      </div>
      <div className="research-scenarios">
        {SCENARIOS.map((scenario) => (
          <article className="research-scenario" key={scenario.code}>
            <div className="research-scenario-head">
              <h3>{scenario.title}</h3>
              <span>{scenario.code}</span>
            </div>
            <p>{scenario.summary}</p>
            <button type="button" onClick={() => onUsePrompt(scenario.prompt)}>
              <span>开始研究</span>
              {scenario.prompt}
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}
