import type { ResearchPlan } from "../lib/researchPresentation";
import { planStatusMeta } from "../lib/researchPresentation";

export function ResearchPlanLedger({ plan }: { plan: ResearchPlan }) {
  return (
    <section className="research-plan-ledger" aria-label="研究计划">
      <header className="research-plan-ledger-head">
        <div>
          <span className="research-kicker">RESEARCH PLAN</span>
          <h3>{plan.goal}</h3>
        </div>
        <div className="research-plan-ledger-meta">
          <span>第 {plan.revision} 版</span>
          <strong>{plan.done}/{plan.total} 已完成</strong>
        </div>
      </header>
      <ol className="research-plan-steps">
        {plan.steps.map((step) => {
          const meta = planStatusMeta(step.status);
          return (
            <li className={`research-plan-step research-plan-step-${meta.tone}`} key={step.seq}>
              <span className="research-plan-mark" aria-hidden="true">{meta.mark}</span>
              <span className="research-plan-seq">{String(step.seq).padStart(2, "0")}</span>
              <span className="research-plan-step-body">
                <strong>{step.action}</strong>
                {step.dep.length > 0 && <small>依赖步骤 {step.dep.join("、")}</small>}
              </span>
              <span className="research-plan-status">{meta.label}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
