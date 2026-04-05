from core.llm_client import LLMClient

client = LLMClient(use_cache=False)

reply = client.chat([
    {"role": "user", "content": "Write 3 lines of pandas code to load a CSV and print df.shape. No explanation."}
])

print("=== LLM replied ===")
print(reply)