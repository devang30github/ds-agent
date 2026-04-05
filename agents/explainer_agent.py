import os
import json
from core.agent_base import AgentBase
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from config import safe_path


class ExplainerAgent(AgentBase):

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        super().__init__(llm, executor, name="ExplainerAgent")

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist writing model explanation code.
Always:
- Load data using variable 'features_path' — never hardcode paths
- Load model using variable 'model_path' — never hardcode paths
- Save plots using variable 'output_dir' — never hardcode paths
- Use matplotlib with Agg backend (no display)
- Print all results with clear labels
- Print SHAP values as JSON
Return code only, no explanation."""

    # ------------------------------------------------------------------
    # Step 1 — generate SHAP explanation code
    # ------------------------------------------------------------------

    def _get_shap_code(self, model_report: dict) -> str:
      is_tree   = any(t in model_report['best_model_name']
                      for t in ['Forest', 'Tree', 'Boost', 'Bagging'])
      is_linear = any(t in model_report['best_model_name']
                      for t in ['Logistic', 'Linear', 'Ridge', 'Lasso'])

      if is_tree:
          explainer_setup = """explainer   = shap.TreeExplainer(model)
  shap_vals   = explainer.shap_values(X)
  # TreeExplainer returns list for classifiers — shape is (n_classes, n_samples, n_features)
  # Safely extract 2D array regardless of output shape
  if isinstance(shap_vals, list):
      shap_values = np.array(shap_vals[1])   # class 1 for binary classification
  else:
      shap_values = np.array(shap_vals)
  # Ensure 2D: (n_samples, n_features)
  if shap_values.ndim == 3:
      shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]"""
      elif is_linear:
          explainer_setup = """explainer   = shap.LinearExplainer(model, X)
  shap_values = np.array(explainer.shap_values(X))
  if shap_values.ndim == 3:
      shap_values = shap_values[1]"""
      else:
          explainer_setup = """background  = shap.kmeans(X, min(10, len(X)))
  explainer   = shap.KernelExplainer(model.predict_proba, background)
  shap_vals   = explainer.shap_values(X.iloc[:30])
  shap_values = np.array(shap_vals[1]) if isinstance(shap_vals, list) else np.array(shap_vals)"""

      reply = self.llm.chat(
          messages=[{
              "role": "user",
              "content": f"""Write Python SHAP explanation code using this exact template.
  Fill in the blanks but do NOT change the SHAP setup block — copy it exactly.

  STRICT RULES:
  - features_path, model_path, output_dir are already defined — use them directly
  - import matplotlib; matplotlib.use('Agg') MUST be first import line
  - NEVER call plt.show()
  - NEVER hardcode any paths

  TEMPLATE — follow this exact structure:
  ```python
  import matplotlib
  matplotlib.use('Agg')
  import pandas as pd
  import numpy as np
  import joblib
  import shap
  import json
  import os
  import matplotlib.pyplot as plt

  df    = pd.read_csv(features_path)
  X     = df[{model_report['feature_cols']}]
  y     = df['{model_report['target_col']}']
  model = joblib.load(model_path)

  # SHAP setup — copy exactly, do not modify
  {explainer_setup}

  # Verify shape matches before plotting
  assert shap_values.shape == X.shape, f"Shape mismatch: shap={{shap_values.shape}} X={{X.shape}}"

  # Feature importance
  importance = dict(zip({model_report['feature_cols']}, np.abs(shap_values).mean(axis=0).tolist()))
  sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
  top_features = list(sorted_imp.keys())[:3]

  print("SHAP_IMPORTANCE:", json.dumps({{k: round(v, 4) for k, v in sorted_imp.items()}}))
  print("TOP_FEATURES:", json.dumps(top_features))

  # Save bar plot
  plt.figure(figsize=(8, 5))
  features_arr = np.array({model_report['feature_cols']})
  importance_arr = np.abs(shap_values).mean(axis=0)
  sorted_idx = np.argsort(importance_arr)
  plt.barh(features_arr[sorted_idx], importance_arr[sorted_idx])
  plt.xlabel('Mean |SHAP value|')
  plt.title('Feature importance')
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=100, bbox_inches='tight')
  plt.close()
  print("PLOT_SAVED: shap_summary.png")
  print("EXPLANATION_DONE: true")
  ```

  Return the complete code above with the SHAP setup block copied exactly."""
          }],
          role="code"
      )
      return self.llm.extract_code(reply)
    # ------------------------------------------------------------------
    # Step 2 — validate code
    # ------------------------------------------------------------------

    def _validate_code(self, code: str) -> tuple[bool, str]:
        issues = []

        if "features_path" not in code:
            issues.append("does not use features_path variable")

        if "model_path" not in code:
            issues.append("does not use model_path variable")

        if "output_dir" not in code:
            issues.append("does not use output_dir variable")

        if "matplotlib.use('Agg')" not in code and 'matplotlib.use("Agg")' not in code:
            issues.append("does not set Agg backend — will crash without display")

        if "SHAP_IMPORTANCE" not in code:
            issues.append("does not print SHAP_IMPORTANCE")

        if issues:
            return False, " | ".join(issues)
        return True, ""

    # ------------------------------------------------------------------
    # Step 3 — fix broken code
    # ------------------------------------------------------------------

    def _fix_shap_code(self, original_code: str, error: str) -> str:
        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""This SHAP explanation code failed. Fix it.

CRITICAL RULES:
- features_path, model_path, output_dir are already defined — use them
- import matplotlib; matplotlib.use('Agg') must come BEFORE pyplot import
- For RandomForestClassifier: shap_values = explainer.shap_values(X) returns a list
  use shap_values[1] for binary classification SHAP values
- Never call plt.show()
- Never hardcode paths

ORIGINAL CODE:
{original_code}

ERROR:
{error}

Return only the fixed code, no explanation."""
            }],
            role="code"
        )
        return self.llm.extract_code(reply)

    # ------------------------------------------------------------------
    # Step 4 — parse SHAP importance from output
    # ------------------------------------------------------------------

    def _parse_shap_output(self, stdout: str) -> dict:
        importance  = {}
        top_features = []

        for line in stdout.splitlines():
            if line.startswith("SHAP_IMPORTANCE:"):
                raw = line.replace("SHAP_IMPORTANCE:", "").strip()
                try:
                    importance = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        import ast
                        importance = ast.literal_eval(raw)
                    except Exception:
                        pass

            if line.startswith("TOP_FEATURES:"):
                raw = line.replace("TOP_FEATURES:", "").strip()
                try:
                    top_features = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        import ast
                        top_features = ast.literal_eval(raw)
                    except Exception:
                        pass

        return {"importance": importance, "top_features": top_features}

    # ------------------------------------------------------------------
    # Step 5 — LLM writes natural language explanation
    # ------------------------------------------------------------------

    def _generate_explanation(self, model_report: dict, shap_data: dict) -> str:
        importance   = shap_data.get("importance", {})
        top_features = shap_data.get("top_features", [])

        # Sort features by importance
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        importance_desc = "\n".join([
            f"  - {feat}: {score:.4f}"
            for feat, score in sorted_feats
        ])

        reply = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"""Write a clear, non-technical explanation of these ML model results.
The audience is a business stakeholder who does not know ML.

MODEL RESULTS:
- Best model:  {model_report['best_model_name']}
- Task:        {model_report['task_type']}
- Target:      {model_report['target_col']}
- Score:       {model_report['best_score']} ({model_report['metric']})
- All models tried: {[(m['name'], round(m['score'], 3)) for m in model_report.get('all_models', [])]}

FEATURE IMPORTANCE (SHAP — higher = more influential):
{importance_desc}

TOP 3 MOST IMPORTANT FEATURES: {top_features}

Write 3 paragraphs:
1. What the model does and how well it performs (avoid jargon)
2. Which factors matter most and why (reference top features by name)
3. What business action could be taken based on these insights

Be specific, concrete, and avoid terms like 'SHAP', 'F1', 'classifier'."""
            }],
            role="reason"
        )
        return reply

    # ------------------------------------------------------------------
    # Main run()
    # ------------------------------------------------------------------

    def run(self, model_report: dict, output_dir: str = "outputs") -> dict:
        os.makedirs(output_dir, exist_ok=True)

        features_path = model_report["features_csv_path"]
        model_path    = model_report["best_model_path"]

        print(f"\n[ExplainerAgent] Explaining: {model_report['best_model_name']}")
        print(f"[ExplainerAgent] Features: {model_report['feature_cols']}")

        # Step 1 — generate SHAP code
        print("[ExplainerAgent] Generating SHAP code...")
        shap_code = self._get_shap_code(model_report)

        # Step 2 — validate
        print("[ExplainerAgent] Validating...")
        is_valid, val_error = self._validate_code(shap_code)

        if not is_valid:
            print(f"[ExplainerAgent] Validation failed: {val_error} — fixing...")
            shap_code = self._fix_shap_code(
                shap_code,
                f"Validation failed: {val_error}"
            )
            is_valid, val_error = self._validate_code(shap_code)
            if not is_valid:
                raise RuntimeError(f"SHAP code still invalid: {val_error}")

        # Step 3 — run
        context = {
            "features_path": features_path,
            "model_path":    model_path,
            "output_dir":    output_dir,
        }
        print("[ExplainerAgent] Running SHAP analysis...")
        result = self._run_code(shap_code, context=context)

        # Step 4 — retry if failed
        if not result["success"]:
            print("[ExplainerAgent] Failed — asking LLM to fix...")
            shap_code = self._fix_shap_code(shap_code, result["stderr"])
            result    = self._run_code(shap_code, context=context)

        if not result["success"]:
            print(f"[ExplainerAgent] SHAP failed — continuing without it.\n{result['stderr']}")
            shap_data = {"importance": {}, "top_features": model_report["feature_cols"][:3]}
        else:
            print("[ExplainerAgent] SHAP done.")
            print(result["stdout"])
            shap_data = self._parse_shap_output(result["stdout"])

        # Step 5 — natural language explanation
        print("[ExplainerAgent] Generating natural language explanation...")
        explanation = self._generate_explanation(model_report, shap_data)

        # Step 6 — check plot saved
        plot_path = safe_path(output_dir, "shap_summary.png")
        plot_saved = os.path.exists(plot_path)

        explain_report = {
            "explanation":       explanation,
            "shap_importance":   shap_data.get("importance", {}),
            "top_features":      shap_data.get("top_features", []),
            "plot_path":         plot_path if plot_saved else None,
            "plot_saved":        plot_saved,
            "best_model_name":   model_report["best_model_name"],
            "best_score":        model_report["best_score"],
            "metric":            model_report["metric"],
            "task_type":         model_report["task_type"],
            "target_col":        model_report["target_col"],
            "success":           True,
        }

        print(f"\n[ExplainerAgent] Done. Plot saved: {plot_saved}")
        print(f"\n{'='*50}")
        print("NATURAL LANGUAGE EXPLANATION:")
        print('='*50)
        print(explanation)

        return explain_report