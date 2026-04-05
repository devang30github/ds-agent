import json
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from agents.eda_agent import EDAAgent

llm      = LLMClient(use_cache=True)
executor = CodeExecutor()
agent    = EDAAgent(llm, executor)

# Use any CSV you have — or create a quick test one
import pandas as pd
import numpy as np

# Create a sample CSV if you don't have one
df = pd.DataFrame({
    "age":        [25, 30, np.nan, 45, 22, 38, np.nan, 51],
    "salary":     [50000, 60000, 75000, 90000, 45000, 80000, 55000, 95000],
    "department": ["eng", "eng", "hr", "eng", "hr", "sales", "sales", np.nan],
    "churn":      [0, 1, 0, 0, 1, 1, 0, 1],
})
df.to_csv("uploads/test.csv", index=False)
print("Sample CSV created at uploads/test.csv")

# Run EDA agent
profile = agent.run(
    csv_path    = "uploads/test.csv",
    user_prompt = "predict customer churn"
)

print("\n=== EDA PROFILE ===")
print(json.dumps({k: v for k, v in profile.items() if k != "raw_profile_output"}, indent=2))