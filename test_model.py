import json
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from agents.eda_agent import EDAAgent
from agents.cleaning_agent import CleaningAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent

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

# Features
print("\n=== FEATURES ===")
feat_agent  = FeatureAgent(llm, executor)
feat_report = feat_agent.run(report)

# Models
print("\n=== MODEL TRAINING ===")
model_agent  = ModelAgent(llm, executor)
model_report = model_agent.run(feat_report)

print("\n=== MODEL REPORT ===")
print(json.dumps({
    k: v for k, v in model_report.items()
    if k != "training_output"
}, indent=2))