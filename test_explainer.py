import json
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from agents.eda_agent import EDAAgent
from agents.cleaning_agent import CleaningAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.explainer_agent import ExplainerAgent

llm      = LLMClient(use_cache=True)
executor = CodeExecutor()

print("=== EDA ===")
eda     = EDAAgent(llm, executor)
profile = eda.run("uploads/test.csv", "predict customer churn")

print("\n=== CLEANING ===")
cleaner = CleaningAgent(llm, executor)
report  = cleaner.run(profile)

print("\n=== FEATURES ===")
feat_agent  = FeatureAgent(llm, executor)
feat_report = feat_agent.run(report)

print("\n=== MODEL TRAINING ===")
model_agent  = ModelAgent(llm, executor)
model_report = model_agent.run(feat_report)

print("\n=== EXPLANATION ===")
explainer      = ExplainerAgent(llm, executor)
explain_report = explainer.run(model_report)

print("\n=== EXPLAIN REPORT ===")
print(json.dumps({
    k: v for k, v in explain_report.items()
    if k != "explanation"
}, indent=2))