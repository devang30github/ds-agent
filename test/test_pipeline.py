import json
from agents.orchestrator import Orchestrator

orchestrator = Orchestrator(use_cache=True)

report = orchestrator.run(
    csv_path    = "uploads/test.csv",
    user_prompt = "predict customer churn",
    output_dir  = "outputs",
)

print("\n=== FINAL REPORT (summary) ===")
print(json.dumps({
    k: v for k, v in report.items()
    if k not in ("explanation", "pipeline_log", "shap_importance")
}, indent=2))