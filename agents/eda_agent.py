import json
from core.agent_base import AgentBase
from core.llm_client import LLMClient
from core.executor import CodeExecutor


class EDAAgent(AgentBase):

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        super().__init__(llm, executor, name="EDAAgent")

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist performing exploratory data analysis.
You write clean, concise pandas code to profile datasets.
When writing code always:
- Use 'csv_path' as the variable for the file path (it is pre-defined)
- Print all results clearly with labels
- Never use plt.show() or any display functions
- Handle errors gracefully with try/except
When asked for JSON, return only valid JSON with no markdown fences or explanation."""

    # ------------------------------------------------------------------
    # Step 1 — basic profile
    # ------------------------------------------------------------------

    def _get_basic_profile_code(self) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": """Write Python code to profile a CSV file loaded from variable 'csv_path'.
Print the following with clear labels:
1. SHAPE: number of rows and columns
2. COLUMNS: list of all column names
3. DTYPES: each column name and its dtype
4. MISSING: count and percentage of missing values per column (only columns with missing values)
5. NUMERIC_STATS: describe() output for numeric columns
6. SAMPLE: first 3 rows as a string
Code only, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 2 — target column analysis
    # ------------------------------------------------------------------

    def _get_target_analysis_code(self, target_col: str, is_classification: bool) -> str:
        task_type = "classification" if is_classification else "regression"
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write Python code to analyze the target column '{target_col}' in a dataframe loaded from 'csv_path'.
This is a {task_type} task.
Print:
1. TARGET_DIST: value counts if classification, basic stats if regression
2. TARGET_NULLS: number of null values in target column
3. CLASS_BALANCE: class percentages if classification (skip if regression)
Code only, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 3 — LLM reasons about the raw output
    # ------------------------------------------------------------------

    def _analyze_profile(
        self,
        raw_output: str,
        target_output: str,
        user_prompt: str
    ) -> dict:
        reply = self.llm.chat_json(
            messages=[{
                "role": "user",
                "content": f"""Based on this dataset profile output, return a JSON analysis.

USER GOAL: {user_prompt}

PROFILE OUTPUT:
{raw_output}

TARGET ANALYSIS:
{target_output}

Return this exact JSON structure:
{{
  "n_rows": <int>,
  "n_cols": <int>,
  "columns": ["col1", "col2", ...],
  "numeric_cols": ["col1", ...],
  "categorical_cols": ["col1", ...],
  "target_col": "<column name>",
  "task_type": "classification" or "regression",
  "missing_cols": [{{"col": "name", "pct": 0.0}}],
  "high_missing_cols": ["cols with >40% missing"],
  "issues": ["list of data quality issues found"],
  "recommendations": ["list of cleaning steps recommended"]
}}"""
            }],
            role="reason"
        )
        return reply

    # ------------------------------------------------------------------
    # Main run() — orchestrates all steps
    # ------------------------------------------------------------------

    def run(self, csv_path: str, user_prompt: str) -> dict:
        """
        Full EDA pipeline.
        Returns a structured profile dict that downstream agents will use.
        """
        print(f"\n[EDAAgent] Starting analysis on: {csv_path}")
        print(f"[EDAAgent] User goal: {user_prompt}")

        # Step 1 — detect task type from user prompt first
        task_hint = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Given this user goal: '{user_prompt}'
Is this a classification task or regression task?
Also what is the likely target column name based on the goal?
Reply in JSON only: {{"task_type": "classification|regression", "target_col": "column_name_guess"}}"""
            }],
            role="reason"
        )
        task_hint_json = json.loads(
            self.llm.extract_code(task_hint) if "```" in task_hint else task_hint
        )
        is_classification = task_hint_json.get("task_type") == "classification"
        target_guess      = task_hint_json.get("target_col", "target").lower()
        
        print(f"[EDAAgent] Detected task: {task_hint_json['task_type']}, target guess: {target_guess}")

        # Step 2 — generate and run basic profile code
        print("[EDAAgent] Running basic profile...")
        profile_code = self._get_basic_profile_code()
        profile_result = self._run_code(profile_code, context={"csv_path": csv_path})

        if not profile_result["success"]:
            raise RuntimeError(f"EDA profiling failed:\n{profile_result['stderr']}")

        print("[EDAAgent] Basic profile done.")

        # Step 3 — run target analysis
        print(f"[EDAAgent] Analyzing target column: {target_guess}")
        target_code   = self._get_target_analysis_code(target_guess, is_classification)
        target_result = self._run_code(target_code, context={"csv_path": csv_path})

        # Step 4 — LLM reasons about the real output
        print("[EDAAgent] Reasoning about profile...")
        profile = self._analyze_profile(
            raw_output    = profile_result["output"],
            target_output = target_result["output"],
            user_prompt   = user_prompt
        )

        # Attach raw output for downstream agents to use
        profile["raw_profile_output"] = profile_result["output"]
        profile["csv_path"]           = csv_path

        print(f"[EDAAgent] Done. Found {profile.get('n_rows')} rows, "
              f"{profile.get('n_cols')} cols, "
              f"{len(profile.get('issues', []))} issues.")

        return profile