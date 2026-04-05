from core.llm_client import LLMClient
from core.executor import CodeExecutor

class AgentBase:
    """
    Base class for all agents.
    Each agent has: a system prompt, access to the LLM, access to the executor,
    and a run() method that subclasses implement.
    """

    def __init__(self, llm: LLMClient, executor: CodeExecutor, name: str):
        self.llm      = llm
        self.executor = executor
        self.name     = name
        self.history  = []    # conversation history this agent maintains

    def _say(self, content: str) -> str:
        """Send a user message, get assistant reply, update history."""
        self.history.append({"role": "user", "content": content})
        reply = self.llm.chat(self.history, system=self.system_prompt)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _run_code(self, code: str, context: dict | None = None) -> dict:
        """Execute code and return result dict."""
        return self.executor.run(code, context)

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError