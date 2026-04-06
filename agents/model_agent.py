import os
import json
from core.agent_base import AgentBase
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from config import safe_path


class ModelAgent(AgentBase):

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        super().__init__(llm, executor, name="ModelAgent")

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist writing scikit-learn model training code.
Always:
- Load data using variable 'features_path' — never hardcode any file path
- Save results using variable 'output_dir' — never hardcode any path
- Print metrics clearly with labels
- Print model comparison as JSON at the end
- Never use plt.show()
Return code only, no explanation."""

    # ------------------------------------------------------------------
    # Step 1 — decide which models to try
    # ------------------------------------------------------------------
    '''
    def _plan_models(self, feature_report: dict) -> dict:
        reply = self.llm.chat_json(
            messages=[{
                "role": "user",
                "content": f"""You are selecting ML models for a pipeline.

TASK:          {feature_report['task_type']}
FEATURES:      {feature_report['feature_cols']}
TARGET:        {feature_report['target_col']}
N ROWS:        {feature_report['n_rows']}
N FEATURES:    {len(feature_report['feature_cols'])}

Select 3 appropriate sklearn models and return a JSON plan:
{{
  "models": [
    {{
      "name": "LogisticRegression",
      "class": "sklearn.linear_model.LogisticRegression",
      "params": {{"max_iter": 1000}},
      "reason": "good baseline for classification"
    }},
    ...
  ],
  "test_size": 0.2,
  "random_state": 42,
  "metric": "accuracy" or "f1" for classification, "rmse" or "r2" for regression,
  "reasoning": "why these models were chosen"
}}

Rules:
- Always include a simple baseline (LogisticRegression or LinearRegression)
- Always include a tree-based model (RandomForest)
- Add one more appropriate model based on dataset size and task
- For small datasets (<100 rows) use simple models, avoid heavy ones
- Never use neural networks or XGBoost (not in base sklearn)"""
            }],
            role="reason"
        )
        return reply
    '''
    def _plan_models(self, feature_report: dict) -> dict:
        """Ask reasoning model to decide which models to train."""

        DEFAULT_CLASSIFICATION_PLAN = {
            "models": [
                {"name": "LogisticRegression",     "class": "sklearn.linear_model.LogisticRegression",    "params": {"max_iter": 1000}, "reason": "linear baseline"},
                {"name": "RandomForestClassifier", "class": "sklearn.ensemble.RandomForestClassifier",    "params": {"random_state": 42}, "reason": "handles non-linear patterns"},
                {"name": "GradientBoostingClassifier", "class": "sklearn.ensemble.GradientBoostingClassifier", "params": {"random_state": 42}, "reason": "strong ensemble model"},
            ],
            "test_size":    0.2,
            "random_state": 42,
            "metric":       "f1",
            "reasoning":    "Default classification plan",
        }

        DEFAULT_REGRESSION_PLAN = {
            "models": [
                {"name": "LinearRegression",  "class": "sklearn.linear_model.LinearRegression",  "params": {}, "reason": "linear baseline"},
                {"name": "RandomForestRegressor", "class": "sklearn.ensemble.RandomForestRegressor", "params": {"random_state": 42}, "reason": "handles non-linear patterns"},
                {"name": "GradientBoostingRegressor", "class": "sklearn.ensemble.GradientBoostingRegressor", "params": {"random_state": 42}, "reason": "strong ensemble model"},
            ],
            "test_size":    0.2,
            "random_state": 42,
            "metric":       "r2",
            "reasoning":    "Default regression plan",
        }

        default = (DEFAULT_CLASSIFICATION_PLAN
                if feature_report["task_type"] == "classification"
                else DEFAULT_REGRESSION_PLAN)

        try:
            reply = self.llm.chat_json(
                messages=[{
                    "role": "user",
                    "content": f"""You are selecting ML models for a pipeline.

    TASK:       {feature_report['task_type']}
    FEATURES:   {feature_report['feature_cols']}
    TARGET:     {feature_report['target_col']}
    N ROWS:     {feature_report['n_rows']}
    N FEATURES: {len(feature_report['feature_cols'])}

    Select 3 appropriate sklearn models and return JSON only:
    {{
    "models": [
        {{
        "name": "LogisticRegression",
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {{"max_iter": 1000}},
        "reason": "good baseline"
        }}
    ],
    "test_size": 0.2,
    "random_state": 42,
    "metric": "f1",
    "reasoning": "why these models"
    }}

    Rules:
    - Always include a linear baseline and RandomForest
    - For small datasets avoid heavy models
    - metric: f1 or accuracy for classification, r2 or rmse for regression
    - Return valid JSON only, no explanation"""
                }],
                role="reason"
            )

            # Validate reply has required keys
            if not reply or not isinstance(reply, dict):
                raise ValueError("Empty or non-dict reply")
            if "models" not in reply or not reply["models"]:
                raise ValueError("No models in plan")

            return reply

        except Exception as e:
            print(f"[ModelAgent] Plan parsing failed: {e} — using default plan")
            return default
    # ------------------------------------------------------------------
    # Step 2 — generate training code
    # ------------------------------------------------------------------

    def _get_training_code(self, feature_report: dict, plan: dict) -> str:
        models_desc = "\n".join([
            f"  - {m['name']}: {m['class']} with params {m['params']}"
            for m in plan["models"]
        ])

        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write Python ML training code.

STRICT RULES:
- Load data using: df = pd.read_csv(features_path)   ← variable already defined
- Save best model using: joblib.dump(best_model, best_model_path)  ← variable already defined
- NEVER hardcode any file paths

TASK:       {feature_report['task_type']}
FEATURES:   {feature_report['feature_cols']}
TARGET:     {feature_report['target_col']}
TEST SIZE:  {plan.get('test_size', 0.2)}
RANDOM STATE: {plan.get('random_state', 42)}
METRIC:     {plan.get('metric', 'accuracy')}

MODELS TO TRAIN:
{models_desc}

REQUIRED STEPS IN ORDER:
1. import pandas, numpy, joblib, sklearn modules
2. df = pd.read_csv(features_path)
3. X = df[{feature_report['feature_cols']}]
4. y = df['{feature_report['target_col']}']
5. train_test_split with test_size={plan.get('test_size', 0.2)}, random_state={plan.get('random_state', 42)}
6. For each model:
   a. Fit on train set
   b. Predict on test set
   c. Compute {plan.get('metric', 'accuracy')} score
   d. Print: MODEL: <name> | SCORE: <score>
7. Pick best model by highest score
8. Print: BEST_MODEL: <name>
9. Print: BEST_SCORE: <score>
10. Save best model: joblib.dump(best_model, best_model_path)
11. Build results dict and print using json.dumps — NEVER use f-strings for JSON:
    import json
    results_dict = {{
        "models": [{{"name": name, "score": round(score, 6)}} for name, score in model_scores.items()],
        "best": best_model_name,
        "metric": "{plan.get('metric', 'accuracy')}"
    }}
    print("RESULTS_JSON:", json.dumps(results_dict))
    
Code only, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 3 — validate code
    # ------------------------------------------------------------------

    def _validate_code(self, code: str) -> tuple[bool, str]:
        issues = []

        if "features_path" not in code:
            issues.append("does not use features_path variable")

        if "best_model_path" not in code:
            issues.append("does not use best_model_path variable")

        if "joblib.dump" not in code:
            issues.append("does not save model with joblib.dump")

        if "RESULTS_JSON" not in code:
            issues.append("does not print RESULTS_JSON")

        if issues:
            return False, " | ".join(issues)
        return True, ""

    # ------------------------------------------------------------------
    # Step 4 — fix broken code
    # ------------------------------------------------------------------
    def _fix_training_code(self, original_code: str, error: str) -> str:
        target_hint = ""
        if "continuous" in error and "classifier" in error:
            target_hint = """
    SPECIFIC FIX: Add y = y.astype(int) right after loading y."""

        fstring_hint = ""
        if "backslash" in error or "f-string" in error:
            fstring_hint = """
    SPECIFIC FIX: You used backslashes inside an f-string expression — illegal in Python.
    To print RESULTS_JSON never use nested f-strings.
    Instead do this:
        import json
        results_dict = {
            "models": [{"name": n, "score": s} for n, s in scores.items()],
            "best": best_name,
            "metric": "accuracy"
        }
        print("RESULTS_JSON:", json.dumps(results_dict))
    """

        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""This ML training code failed. Fix it.
    {target_hint}
    {fstring_hint}
    CRITICAL RULES:
    - Load using variable:       features_path   (already defined)
    - Save model using variable: best_model_path (already defined)
    - Must print: RESULTS_JSON: {{...}} using json.dumps, never manual f-strings
    - Never hardcode any file paths
    - For classification: always cast y = y.astype(int) after loading

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
    # Step 5 — parse results from stdout
    # ------------------------------------------------------------------
    def _parse_results(self, stdout: str) -> dict:
        """Extract RESULTS_JSON from printed output. Handles single-quote Python dicts."""
        for line in stdout.splitlines():
            if line.startswith("RESULTS_JSON:"):
                raw = line.replace("RESULTS_JSON:", "").strip()

                # Try valid JSON first
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass

                # Fallback — Python dict with single quotes
                try:
                    import ast
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, dict):
                        return parsed
                except (ValueError, SyntaxError):
                    pass

                print(f"[ModelAgent] Warning: could not parse RESULTS_JSON: {raw}")

        # Last resort — scrape individual lines
        models  = []
        best_name  = None
        best_score = 0.0

        for line in stdout.splitlines():
            if line.startswith("MODEL:"):
                # MODEL: LogisticRegression | SCORE: 0.87
                parts = line.replace("MODEL:", "").split("|")
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        score = float(parts[1].replace("SCORE:", "").strip())
                        models.append({"name": name, "score": score})
                        if score > best_score:
                            best_score = score
                            best_name  = name
                    except ValueError:
                        pass

        if best_name:
            return {"models": models, "best": best_name, "score": best_score, "metric": "accuracy"}

        raise RuntimeError(f"Could not parse results from output:\n{stdout}")
    # ------------------------------------------------------------------
    # Main run()
    # ------------------------------------------------------------------

    def run(self, feature_report: dict, output_dir: str = "outputs") -> dict:
        os.makedirs(output_dir, exist_ok=True)

        features_path  = feature_report["features_csv_path"]
        best_model_path = safe_path(output_dir, "best_model.joblib")

        print(f"\n[ModelAgent] Starting model training: {features_path}")
        print(f"[ModelAgent] Task: {feature_report['task_type']}")
        print(f"[ModelAgent] Target: {feature_report['target_col']}")
        print(f"[ModelAgent] Features: {feature_report['feature_cols']}")

        # Step 1 — plan models
        print("[ModelAgent] Planning models...")
        plan = self._plan_models(feature_report)
        print(f"[ModelAgent] Reasoning: {plan.get('reasoning', '')}")
        print(f"[ModelAgent] Models: {[m['name'] for m in plan['models']]}")
        print(f"[ModelAgent] Metric: {plan.get('metric', 'accuracy')}")

        # Step 2 — generate training code
        print("[ModelAgent] Generating training code...")
        training_code = self._get_training_code(feature_report, plan)

        # Step 3 — validate
        print("[ModelAgent] Validating generated code...")
        is_valid, val_error = self._validate_code(training_code)

        if not is_valid:
            print(f"[ModelAgent] Validation failed: {val_error} — fixing...")
            training_code = self._fix_training_code(
                training_code,
                f"Validation failed: {val_error}. "
                f"Use 'features_path' to load data, 'best_model_path' to save model, "
                f"and print RESULTS_JSON at the end."
            )
            is_valid, val_error = self._validate_code(training_code)
            if not is_valid:
                raise RuntimeError(f"Code still invalid after fix: {val_error}")

        # Step 4 — run
        context = {
            "features_path":  features_path,
            "best_model_path": best_model_path,
        }
        print("[ModelAgent] Running training...")
        result = self._run_code(training_code, context=context)

        # Step 5 — retry if failed
        if not result["success"]:
            print("[ModelAgent] Training failed — asking LLM to fix...")
            training_code = self._fix_training_code(training_code, result["stderr"])
            result        = self._run_code(training_code, context=context)

        if not result["success"]:
            raise RuntimeError(f"Model training failed:\n{result['stderr']}")

        print("[ModelAgent] Training done.")
        print(result["stdout"])

        # Step 6 — parse results
        results = self._parse_results(result["stdout"])

        # Step 7 — verify model file saved
        model_saved = os.path.exists(best_model_path)
        if not model_saved:
            print("[ModelAgent] Warning: model file not found at", best_model_path)

        model_report = {
            "best_model_name":  results.get("best"),
            "best_score": results.get("score") or results.get("best_score") or (
                max((m["score"] for m in results.get("models", [])), default=None)),
            "metric": results.get("metric") or plan.get("metric", "accuracy"),
            "all_models":       results.get("models", []),
            "best_model_path":  best_model_path,
            "model_saved":      model_saved,
            "task_type":        feature_report["task_type"],
            "target_col":       feature_report["target_col"],
            "feature_cols":     feature_report["feature_cols"],
            "features_csv_path": features_path,
            "training_output":  result["stdout"],
            "plan":             plan,
            "success":          True,
        }

        print(f"\n[ModelAgent] Best model: {model_report['best_model_name']}")
        print(f"[ModelAgent] Score ({model_report['metric']}): {model_report['best_score']}")
        return model_report