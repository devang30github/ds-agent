import os
import re
from core.agent_base import AgentBase
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from config import safe_path

class CleaningAgent(AgentBase):

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        super().__init__(llm, executor, name="CleaningAgent")

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist writing pandas data cleaning code.
You write clean, production-quality Python code.
Always:
- Load data using the variable 'csv_path' — never hardcode any file path
- Save cleaned dataframe using the variable 'output_path' — never hardcode any file path
- Use df.copy() to avoid modifying the original
- Print each cleaning step with a clear label as you do it
- Print final shape after cleaning
- Never use plt.show() or any display functions
- Handle errors with try/except where appropriate
Return code only, no explanation."""

    # ------------------------------------------------------------------
    # Step 1 — generate cleaning code from profile
    # ------------------------------------------------------------------

    def _get_cleaning_code(self, profile: dict) -> str:
        missing_info = "\n".join([
            f"  - {m['col']}: {m['pct']}% missing"
            for m in profile.get("missing_cols", [])
        ]) or "  None"

        high_missing  = profile.get("high_missing_cols", [])
        numeric_cols  = profile.get("numeric_cols", [])
        cat_cols      = profile.get("categorical_cols", [])
        target_col    = profile.get("target_col", "")
        issues        = "\n".join([f"  - {i}" for i in profile.get("issues", [])]) or "  None"
        recs          = "\n".join([f"  - {r}" for r in profile.get("recommendations", [])]) or "  None"

        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write Python cleaning code for this dataset.

STRICT RULES — these variables are already defined, use them exactly:
  csv_path    = "{profile['csv_path']}"   <- use this to load data
  output_path = (defined externally)      <- use this to save output

DO NOT hardcode any file paths.
DO NOT write pd.read_csv('input.csv') or any literal path string.
ALWAYS write: df = pd.read_csv(csv_path)
ALWAYS write: df.to_csv(output_path, index=False)

DATASET INFO:
- Shape: {profile['n_rows']} rows x {profile['n_cols']} cols
- Numeric columns: {numeric_cols}
- Categorical columns: {cat_cols}
- Target column: {target_col} (DO NOT modify or impute this column)

MISSING VALUES:
{missing_info}

HIGH MISSING COLS (>40%, drop these entirely): {high_missing}

ISSUES FOUND:
{issues}

RECOMMENDED STEPS:
{recs}

CLEANING RULES TO FOLLOW IN ORDER:
1. df = pd.read_csv(csv_path)
2. df = df.copy()
3. Drop any columns in {high_missing} if they exist
4. Impute numeric missing values with median (skip target column '{target_col}')
5. Impute categorical missing values with mode (skip target column '{target_col}')
6. Remove duplicate rows
7. Strip whitespace from all string columns
8. df.to_csv(output_path, index=False)
9. Print each step with a clear label
10. Print final shape as: FINAL SHAPE: (rows, cols)

Code only, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 2 — verify cleaned output
    # ------------------------------------------------------------------

    def _get_verification_code(self, output_path: str) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write Python code to verify a cleaned CSV.

The file is at: output_path = "{output_path}"
Use: df = pd.read_csv(output_path)

Print:
1. CLEANED_SHAPE: rows and columns
2. REMAINING_NULLS: columns still having null values and their counts
3. DUPLICATES: number of duplicate rows remaining
4. DTYPES: each column name and its dtype

Code only, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 3 — fix broken code
    # ------------------------------------------------------------------

    def _fix_cleaning_code(self, original_code: str, error: str) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""This cleaning code failed. Fix it.

CRITICAL RULES:
- Load data using variable: csv_path  (already defined, do not reassign)
- Save output using variable: output_path  (already defined, do not reassign)
- Never hardcode any file path strings

ORIGINAL CODE:
{original_code}

ERROR:
{error}

Return only the fixed Python code, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 4 — validate code uses correct variables
    # ------------------------------------------------------------------

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """
        Checks the generated code actually uses csv_path and output_path.
        Returns (is_valid, error_message).
        """
        issues = []

        if "csv_path" not in code:
            issues.append("does not use csv_path variable")

        if "output_path" not in code:
            issues.append("does not use output_path variable")

        hardcoded = re.findall(r"""pd\.read_csv\(\s*['"][^'"]+['"]\s*\)""", code)
        if hardcoded:
            issues.append(f"hardcoded path found: {hardcoded}")

        if issues:
            return False, " | ".join(issues)
        return True, ""

    # ------------------------------------------------------------------
    # Main run()
    # ------------------------------------------------------------------

    def run(self, profile: dict, output_dir: str = "outputs") -> dict:
        """
        Full cleaning pipeline.
        Returns a cleaning report dict with path to the cleaned CSV.
        """
        csv_path    = profile["csv_path"]
        output_path = safe_path(output_dir, "cleaned.csv")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[CleaningAgent] Starting cleaning: {csv_path}")
        print(f"[CleaningAgent] Output: {output_path}")

        # Step 1 — generate cleaning code
        print("[CleaningAgent] Generating cleaning code...")
        cleaning_code = self._get_cleaning_code(profile)

        # Step 2 — validate before running
        print("[CleaningAgent] Validating generated code...")
        is_valid, val_error = self._validate_code(cleaning_code)

        if not is_valid:
            print(f"[CleaningAgent] Validation failed: {val_error} — fixing...")
            cleaning_code = self._fix_cleaning_code(
                cleaning_code,
                f"Validation failed: {val_error}. "
                f"You MUST use 'csv_path' to load and 'output_path' to save. "
                f"Never hardcode any file paths."
            )
            # Validate once more after fix
            is_valid, val_error = self._validate_code(cleaning_code)
            if not is_valid:
                raise RuntimeError(f"Code still invalid after fix attempt: {val_error}")

        # Step 3 — run it
        context = {"csv_path": csv_path, "output_path": output_path}
        print("[CleaningAgent] Running cleaning code...")
        result = self._run_code(cleaning_code, context=context)

        # Step 4 — if it failed, ask LLM to fix and retry once
        if not result["success"]:
            print("[CleaningAgent] Execution failed — asking LLM to fix...")
            cleaning_code = self._fix_cleaning_code(cleaning_code, result["stderr"])
            result        = self._run_code(cleaning_code, context=context)

        if not result["success"]:
            raise RuntimeError(f"Cleaning failed after retry:\n{result['stderr']}")

        print("[CleaningAgent] Cleaning done.")
        print(result["stdout"])

        # Step 5 — verify the output file actually exists
        cleaned_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if not cleaned_exists:
            raise RuntimeError("Cleaning appeared to succeed but output file is missing or empty.")

        # Step 6 — verify contents
        print("[CleaningAgent] Verifying cleaned file...")
        verify_code   = self._get_verification_code(output_path)
        verify_result = self._run_code(
            verify_code,
            context={"output_path": output_path}
        )
        print(verify_result["stdout"])

        # Step 7 — build and return report
        report = {
            "cleaned_csv_path": output_path,
            "cleaning_code":    cleaning_code,
            "cleaning_output":  result["stdout"],
            "verify_output":    verify_result["stdout"],
            "success":          cleaned_exists,
            "n_rows_original":  profile["n_rows"],
            "numeric_cols":     profile["numeric_cols"],
            "categorical_cols": profile["categorical_cols"],
            "target_col":       profile["target_col"],
            "task_type":        profile["task_type"],
            "columns":          profile["columns"],
        }

        print(f"[CleaningAgent] Done. Cleaned CSV saved to: {output_path}")
        return report