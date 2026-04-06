import json
from core.agent_base import AgentBase
from core.llm_client import LLMClient
from core.executor import CodeExecutor
import re

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
    # -----------------------------------------------------------------

    def _validate_profile_code(self, code: str) -> tuple[bool, str]:
        issues = []

        if "csv_path" not in code:
            issues.append("does not use csv_path variable")

        hardcoded = re.findall(r"""pd\.read_csv\(\s*['"][^'"]+['"]\s*\)""", code)
        if hardcoded:
            issues.append(f"hardcoded path found: {hardcoded}")

        if issues:
            return False, " | ".join(issues)
        return True, ""

    def _fix_profile_code(self, original_code: str, error: str) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""This profiling code failed. Fix it.

    CRITICAL RULES:
    - Load data using variable: csv_path  (already defined, never reassign)
    - NEVER hardcode any file path like 'example.csv' or 'data.csv'
    - Always write: df = pd.read_csv(csv_path)

    ORIGINAL CODE:
    {original_code}

    ERROR:
    {error}

    Return only the fixed Python code, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    def _get_basic_profile_code(self, csv_path: str) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write Python code to profile a CSV file.

    STRICT RULES:
    - The file path is available in variable: csv_path = "{csv_path}"
    - ALWAYS write: df = pd.read_csv(csv_path)
    - NEVER hardcode any file path string
    - NEVER call a wrapper function with a literal path like profile_csv('example.csv')

    Print the following with clear labels:
    1. SHAPE: number of rows and columns
    2. COLUMNS: list of all column names
    3. DTYPES: each column name and its dtype
    4. MISSING: count and percentage of missing values per column
    5. NUMERIC_STATS: describe() output for numeric columns
    6. SAMPLE: first 3 rows as string

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
                "content": f"""Write Python code to analyze the target column '{target_col}'.

    STRICT RULES:
    - Load data using variable: csv_path  (already defined)
    - ALWAYS write: df = pd.read_csv(csv_path)
    - NEVER hardcode any file path

    This is a {task_type} task.
    Print:
    1. TARGET_DIST: value counts if classification, basic stats if regression
    2. TARGET_NULLS: number of null values in target column
    3. CLASS_BALANCE: class percentages if classification

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
        print(f"\n[EDAAgent] Starting analysis on: {csv_path}")
        print(f"[EDAAgent] User goal: {user_prompt}")

        # Step 1 — detect task type
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

        # Step 2 — generate and validate profile code
        print("[EDAAgent] Running basic profile...")
        profile_code = self._get_basic_profile_code(csv_path)

        # Validate
        is_valid, val_error = self._validate_profile_code(profile_code)
        if not is_valid:
            print(f"[EDAAgent] Validation failed: {val_error} — fixing...")
            profile_code = self._fix_profile_code(
                profile_code,
                f"Validation failed: {val_error}. Use csv_path variable, never hardcode paths."
            )
            is_valid, val_error = self._validate_profile_code(profile_code)
            if not is_valid:
                raise RuntimeError(f"Profile code invalid after fix: {val_error}")

        # Run
        profile_result = self._run_code(profile_code, context={"csv_path": csv_path})

        if not profile_result["success"]:
            # Try to fix and retry once
            print("[EDAAgent] Profile failed — fixing...")
            profile_code   = self._fix_profile_code(profile_code, profile_result["stderr"])
            profile_result = self._run_code(profile_code, context={"csv_path": csv_path})

        if not profile_result["success"]:
            raise RuntimeError(f"EDA profiling failed:\n{profile_result['stderr']}")

        print("[EDAAgent] Basic profile done.")

        # Step 3 — target analysis
        print(f"[EDAAgent] Analyzing target column: {target_guess}")
        target_code   = self._get_target_analysis_code(target_guess, is_classification)
        target_result = self._run_code(target_code, context={"csv_path": csv_path})

        # Step 4 — reason about profile
        print("[EDAAgent] Reasoning about profile...")
        profile = self._analyze_profile(
            raw_output    = profile_result["output"],
            target_output = target_result["output"],
            user_prompt   = user_prompt
        )
        if not profile:
            raise RuntimeError("EDA profile analysis returned no result — check LLM response")

        profile["raw_profile_output"] = profile_result["output"]
        profile["csv_path"]           = csv_path

        print(f"[EDAAgent] Done. Found {profile.get('n_rows')} rows, "
            f"{profile.get('n_cols')} cols, "
            f"{len(profile.get('issues', []))} issues.")

        return profile