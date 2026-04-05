import os
import re
from core.agent_base import AgentBase
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from config import safe_path

class FeatureAgent(AgentBase):

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        super().__init__(llm, executor, name="FeatureAgent")

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist writing feature engineering code.
You write clean, production-quality Python using pandas and scikit-learn.
Always:
- Load data using variable 'cleaned_path' — never hardcode any file path
- Save result using variable 'output_path' — never hardcode any file path
- The LAST line of code must always be: df.to_csv(output_path, index=False)
- Never drop or modify the target column
- Print each step with a clear label
- Print final shape and list of all column names
- Never use plt.show() or any display functions
Return code only, no explanation."""

    # ------------------------------------------------------------------
    # Step 1 — LLM decides which features to engineer
    # ------------------------------------------------------------------

    def _plan_features(self, report: dict) -> dict:
        reply = self.llm.chat_json(
            messages=[{
                "role": "user",
                "content": f"""You are planning feature engineering for a machine learning pipeline.

DATASET INFO:
- Numeric columns:     {report['numeric_cols']}
- Categorical columns: {report['categorical_cols']}
- Target column:       {report['target_col']} (never touch this)
- Task type:           {report['task_type']}
- Original rows:       {report['n_rows_original']}

Decide what feature engineering to apply and return a JSON plan:
{{
  "encode_categoricals": ["list of categorical cols to label encode"],
  "scale_numerics": ["list of numeric cols to standard scale (exclude target)"],
  "create_interactions": [
    {{"name": "feat1_x_feat2", "col1": "feat1", "col2": "feat2", "operation": "multiply"}}
  ],
  "drop_cols": ["cols to drop if any (never drop target)"],
  "reasoning": "brief explanation of decisions"
}}

Rules:
- Always encode ALL categorical columns
- Scale numeric columns except the target
- Only create interactions if dataset has more than 100 rows, else empty list
- Drop cols only if they are IDs or clearly irrelevant"""
            }],
            role="reason"
        )
        return reply

    # ------------------------------------------------------------------
    # Step 2 — generate feature engineering code from plan
    # ------------------------------------------------------------------

    def _get_feature_code(self, report: dict, plan: dict) -> str:
        interactions = plan.get("create_interactions", [])
        interaction_desc = "\n".join([
            f"  - Create '{i['name']}' = {i['col1']} {i['operation']} {i['col2']}"
            for i in interactions
        ]) or "  None (dataset too small)"

        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write Python feature engineering code.

STRICT RULES — memorize these before writing a single line:
1. First line after imports:  df = pd.read_csv(cleaned_path)
2. Last line of code:         df.to_csv(output_path, index=False)
3. NEVER hardcode any file path string
4. NEVER drop or modify target column '{report['target_col']}'
5. Variables 'cleaned_path' and 'output_path' are already defined — just use them

FEATURE ENGINEERING STEPS:
1. Load:     df = pd.read_csv(cleaned_path)
2. Encode these categorical columns with LabelEncoder: {plan.get('encode_categoricals', [])}
3. Standard scale these numeric columns (skip target): {plan.get('scale_numerics', [])}
4. Create interaction features:
{interaction_desc}
5. Drop these columns if they exist: {plan.get('drop_cols', [])}
6. Save:     df.to_csv(output_path, index=False)

REQUIRED IMPORTS: pandas, numpy, sklearn.preprocessing

REQUIRED PRINTS (in this order):
- Print each step with a label as you do it
- Print "NEW COLUMNS: [list]" for any new columns added
- Print "FINAL SHAPE: (rows, cols)"
- Print "FEATURES: [list of all final column names]"

Code only, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 3 — validate code uses correct variables
    # ------------------------------------------------------------------

    def _validate_code(self, code: str) -> tuple[bool, str]:
        issues = []

        if "cleaned_path" not in code:
            issues.append("does not use cleaned_path variable")

        if "output_path" not in code:
            issues.append("does not use output_path variable")

        if "to_csv" not in code:
            issues.append("does not call to_csv to save output")

        hardcoded = re.findall(r"""pd\.read_csv\(\s*['"][^'"]+['"]\s*\)""", code)
        if hardcoded:
            issues.append(f"hardcoded path found: {hardcoded}")

        if issues:
            return False, " | ".join(issues)
        return True, ""

    # ------------------------------------------------------------------
    # Step 4 — fix broken code
    # ------------------------------------------------------------------

    def _fix_feature_code(self, original_code: str, error: str) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""This feature engineering code failed. Fix it.

CRITICAL RULES:
- Load using variable:  cleaned_path  (already defined, do not reassign it)
- Save using variable:  output_path   (already defined, do not reassign it)
- The LAST line must be: df.to_csv(output_path, index=False)
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
    # Step 5 — parse final feature list from execution output
    # ------------------------------------------------------------------

    def _parse_features_from_output(self, stdout: str, fallback_cols: list) -> list:
        for line in stdout.splitlines():
            if line.startswith("FEATURES:"):
                raw  = line.replace("FEATURES:", "").strip().strip("[]")
                cols = [c.strip().strip("'\"") for c in raw.split(",")]
                return [c for c in cols if c]
        return fallback_cols

    # ------------------------------------------------------------------
    # Main run()
    # ------------------------------------------------------------------

    def run(self, report: dict, output_dir: str = "outputs") -> dict:
        cleaned_path = report["cleaned_csv_path"]
        output_path  = safe_path(output_dir, "features.csv")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[FeatureAgent] Starting feature engineering: {cleaned_path}")

        # Step 1 — plan
        print("[FeatureAgent] Planning features...")
        plan = self._plan_features(report)
        print(f"[FeatureAgent] Plan: {plan.get('reasoning', '')}")
        print(f"[FeatureAgent] Encode: {plan.get('encode_categoricals', [])}")
        print(f"[FeatureAgent] Scale:  {plan.get('scale_numerics', [])}")
        print(f"[FeatureAgent] Interactions: {len(plan.get('create_interactions', []))}")

        # Step 2 — generate code
        print("[FeatureAgent] Generating feature code...")
        feature_code = self._get_feature_code(report, plan)

        # Step 3 — validate, fix if needed
        print("[FeatureAgent] Validating generated code...")
        is_valid, val_error = self._validate_code(feature_code)

        if not is_valid:
            print(f"[FeatureAgent] Validation failed: {val_error} — fixing...")
            feature_code = self._fix_feature_code(
                feature_code,
                f"Validation failed: {val_error}. "
                f"You MUST use 'cleaned_path' to load, 'output_path' to save, "
                f"and the last line MUST be df.to_csv(output_path, index=False)."
            )
            is_valid, val_error = self._validate_code(feature_code)
            if not is_valid:
                raise RuntimeError(f"Code still invalid after fix: {val_error}")

        # Step 4 — run
        context = {"cleaned_path": cleaned_path, "output_path": output_path}
        print("[FeatureAgent] Running feature code...")
        result = self._run_code(feature_code, context=context)

        # Step 5 — retry if failed
        if not result["success"]:
            print("[FeatureAgent] Execution failed — asking LLM to fix...")
            feature_code = self._fix_feature_code(feature_code, result["stderr"])
            result       = self._run_code(feature_code, context=context)

        if not result["success"]:
            raise RuntimeError(f"Feature engineering failed:\n{result['stderr']}")

        print("[FeatureAgent] Feature engineering done.")
        print(result["stdout"])

        # Step 6 — verify file saved, use fallback if LLM forgot to_csv
        output_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

        if not output_exists:
            print("[FeatureAgent] Output file missing — running fallback save...")
            fallback_code = """
import pandas as pd
df = pd.read_csv(cleaned_path)
df.to_csv(output_path, index=False)
print("FALLBACK SAVE: saved", len(df), "rows to output_path")
"""
            fallback_result = self._run_code(
                fallback_code,
                context={"cleaned_path": cleaned_path, "output_path": output_path}
            )
            print(fallback_result["stdout"])
            output_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

        if not output_exists:
            raise RuntimeError("Output file could not be saved even after fallback.")

        # Step 7 — build feature report
        final_features = self._parse_features_from_output(
            result["stdout"],
            fallback_cols=report["columns"]
        )
        target       = report["target_col"]
        feature_cols = [f for f in final_features if f != target]

        feature_report = {
            "features_csv_path": output_path,
            "feature_cols":      feature_cols,
            "target_col":        target,
            "task_type":         report["task_type"],
            "n_rows":            report["n_rows_original"],
            "encode_plan":       plan.get("encode_categoricals", []),
            "scale_plan":        plan.get("scale_numerics", []),
            "interactions_plan": plan.get("create_interactions", []),
            "feature_code":      feature_code,
            "feature_output":    result["stdout"],
            "success":           output_exists,
        }

        print(f"[FeatureAgent] Done. {len(feature_cols)} features ready.")
        print(f"[FeatureAgent] Feature columns: {feature_cols}")
        return feature_report