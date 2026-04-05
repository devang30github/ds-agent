import json
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from agents.eda_agent import EDAAgent
from agents.cleaning_agent import CleaningAgent
from agents.feature_agent import FeatureAgent

llm      = LLMClient(use_cache=True)
executor = CodeExecutor()

# EDA
print("=== EDA ===")
eda     = EDAAgent(llm, executor)
profile = eda.run("uploads/test.csv", "predict customer churn")

# Cleaning
print("\n=== CLEANING ===")
cleaner = CleaningAgent(llm, executor)
report  = cleaner.run(profile)

# Feature Engineering
print("\n=== FEATURE ENGINEERING ===")
feat_agent    = FeatureAgent(llm, executor)
feat_report   = feat_agent.run(report)

print("\n=== FEATURE REPORT ===")
print(json.dumps({
    k: v for k, v in feat_report.items()
    if k not in ("feature_code", "feature_output")
}, indent=2))