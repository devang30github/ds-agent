import json
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from agents.eda_agent import EDAAgent
from agents.cleaning_agent import CleaningAgent

llm      = LLMClient(use_cache=True)
executor = CodeExecutor()

# Run EDA first to get profile
eda     = EDAAgent(llm, executor)
profile = eda.run(
    csv_path    = "uploads/test.csv",
    user_prompt = "predict customer churn"
)

print("\n" + "="*50)

# Run Cleaning
cleaner = CleaningAgent(llm, executor)
report  = cleaner.run(profile)

print("\n=== CLEANING REPORT ===")
print(json.dumps({
    k: v for k, v in report.items()
    if k not in ("cleaning_code", "cleaning_output", "verify_output")
}, indent=2))

print("\n=== CLEANING OUTPUT ===")
print(report["cleaning_output"])

print("\n=== VERIFICATION ===")
print(report["verify_output"])