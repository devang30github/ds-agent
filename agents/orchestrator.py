import os
import json
import time
from core.llm_client import LLMClient
from core.executor import CodeExecutor
from agents.eda_agent import EDAAgent
from agents.cleaning_agent import CleaningAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.explainer_agent import ExplainerAgent
from config import safe_path


class Orchestrator:
    """
    Runs the full data science pipeline end to end.
    EDA → Cleaning → Feature Engineering → Model Training → Explanation
    """

    def __init__(self, use_cache: bool = True):
        self.llm      = LLMClient(use_cache=use_cache)
        self.executor = CodeExecutor()

        self.eda_agent      = EDAAgent(self.llm, self.executor)
        self.cleaning_agent = CleaningAgent(self.llm, self.executor)
        self.feature_agent  = FeatureAgent(self.llm, self.executor)
        self.model_agent    = ModelAgent(self.llm, self.executor)
        self.explainer      = ExplainerAgent(self.llm, self.executor)

    def run(
        self,
        csv_path: str,
        user_prompt: str,
        output_dir: str = "outputs",
    ) -> dict:
        """
        Full pipeline. Returns a final report dict with everything.
        """
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()

        print("\n" + "="*60)
        print("  AUTONOMOUS DATA SCIENTIST")
        print("="*60)
        print(f"  CSV:    {csv_path}")
        print(f"  Goal:   {user_prompt}")
        print(f"  Output: {output_dir}")
        print("="*60)

        pipeline_log = []

        def log_step(step: str, status: str, detail: str = ""):
            entry = {"step": step, "status": status, "detail": detail}
            pipeline_log.append(entry)
            icon = "✓" if status == "done" else "✗" if status == "failed" else "..."
            print(f"\n[{icon}] {step}: {detail}")

        try:
            # ----------------------------------------------------------
            # Step 1 — EDA
            # ----------------------------------------------------------
            log_step("EDA", "running", "profiling dataset")
            profile = self.eda_agent.run(csv_path, user_prompt)
            log_step("EDA", "done",
                     f"{profile['n_rows']} rows, {profile['n_cols']} cols, "
                     f"{len(profile.get('issues', []))} issues")

            # ----------------------------------------------------------
            # Step 2 — Cleaning
            # ----------------------------------------------------------
            log_step("Cleaning", "running", "handling missing values and duplicates")
            cleaning_report = self.cleaning_agent.run(profile, output_dir)
            log_step("Cleaning", "done",
                     f"saved to {cleaning_report['cleaned_csv_path']}")

            # ----------------------------------------------------------
            # Step 3 — Feature Engineering
            # ----------------------------------------------------------
            log_step("Feature Engineering", "running", "encoding, scaling, interactions")
            feature_report = self.feature_agent.run(cleaning_report, output_dir)
            log_step("Feature Engineering", "done",
                     f"{len(feature_report['feature_cols'])} features ready")

            # ----------------------------------------------------------
            # Step 4 — Model Training
            # ----------------------------------------------------------
            log_step("Model Training", "running", "training and comparing models")
            model_report = self.model_agent.run(feature_report, output_dir)
            log_step("Model Training", "done",
                     f"best: {model_report['best_model_name']} "
                     f"({model_report['metric']}={round(model_report['best_score'] or 0, 4)})")

            # ----------------------------------------------------------
            # Step 5 — Explanation
            # ----------------------------------------------------------
            log_step("Explanation", "running", "SHAP analysis + natural language report")
            explain_report = self.explainer.run(model_report, output_dir)
            log_step("Explanation", "done",
                     f"top features: {explain_report['top_features']}")

            # ----------------------------------------------------------
            # Final report
            # ----------------------------------------------------------
            elapsed = round(time.time() - start_time, 1)

            final_report = {
                "status":          "success",
                "csv_path":        csv_path,
                "user_prompt":     user_prompt,
                "elapsed_seconds": elapsed,
                "pipeline_log":    pipeline_log,

                # EDA
                "n_rows":          profile["n_rows"],
                "n_cols":          profile["n_cols"],
                "task_type":       profile["task_type"],
                "target_col":      profile["target_col"],
                "issues":          profile.get("issues", []),

                # Features
                "feature_cols":    feature_report["feature_cols"],
                "n_features":      len(feature_report["feature_cols"]),

                # Model
                "best_model":      model_report["best_model_name"],
                "best_score":      model_report["best_score"],
                "metric":          model_report["metric"],
                "all_models":      model_report["all_models"],
                "model_path":      model_report["best_model_path"],

                # Explanation
                "explanation":     explain_report["explanation"],
                "shap_importance": explain_report["shap_importance"],
                "top_features":    explain_report["top_features"],
                "plot_path":       explain_report["plot_path"],

                # Output files
                "output_files": {
                    "cleaned_csv":  cleaning_report["cleaned_csv_path"],
                    "features_csv": feature_report["features_csv_path"],
                    "best_model":   model_report["best_model_path"],
                    "shap_plot":    explain_report.get("plot_path"),
                    "report_json":  safe_path(output_dir, "report.json"),
                }
            }

            # Save report to disk
            report_path = safe_path(output_dir, "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, default=str)

            self._print_summary(final_report)
            return final_report

        except Exception as e:
            elapsed = round(time.time() - start_time, 1)
            log_step("Pipeline", "failed", str(e))

            error_report = {
                "status":          "failed",
                "error":           str(e),
                "csv_path":        csv_path,
                "user_prompt":     user_prompt,
                "elapsed_seconds": elapsed,
                "pipeline_log":    pipeline_log,
            }

            # Save error report
            report_path = safe_path(output_dir, "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(error_report, f, indent=2)

            print(f"\n[!] Pipeline failed after {elapsed}s: {e}")
            raise

    def _print_summary(self, report: dict):
        print("\n" + "="*60)
        print("  PIPELINE COMPLETE")
        print("="*60)
        print(f"  Status:      {report['status']}")
        print(f"  Time:        {report['elapsed_seconds']}s")
        print(f"  Task:        {report['task_type']} → {report['target_col']}")
        print(f"  Rows:        {report['n_rows']}")
        print(f"  Features:    {report['n_features']} ({report['feature_cols']})")
        print(f"  Best model:  {report['best_model']}")
        print(f"  Score:       {report['metric']} = {round(report['best_score'] or 0, 4)}")
        print(f"  Top factors: {report['top_features']}")
        print(f"  Output dir:  {list(report['output_files'].values())}")
        print("="*60)
        print("\n  EXPLANATION:")
        print("  " + report["explanation"].replace("\n", "\n  "))
        print("="*60)