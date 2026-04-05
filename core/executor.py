import subprocess
import sys
import tempfile
import os
import re


class CodeExecutor:
    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def _sanitize_code(self, code: str) -> str:
        """
        Add UTF-8 encoding declaration and strip non-ASCII characters
        that LLMs sometimes put in comments (smart quotes, em dashes etc).
        """
        # Add encoding header
        header = "# -*- coding: utf-8 -*-\n"

        # Replace common smart punctuation with ASCII equivalents
        replacements = {
            "\u2014": "--",   # em dash —
            "\u2013": "-",    # en dash –
            "\u2018": "'",    # left single quote '
            "\u2019": "'",    # right single quote '
            "\u201c": '"',    # left double quote "
            "\u201d": '"',    # right double quote "
            "\u2026": "...",  # ellipsis …
            "\u00a0": " ",    # non-breaking space
        }
        for char, replacement in replacements.items():
            code = code.replace(char, replacement)

        # Strip any remaining non-ASCII from comment lines only
        # (preserve string content which may legitimately have unicode)
        lines = []
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Comment line — safe to strip non-ASCII
                line = line.encode("ascii", errors="ignore").decode("ascii")
            lines.append(line)

        return header + "\n".join(lines)

    def run(self, code: str, context: dict | None = None) -> dict:
        preamble = ""
        if context:
            for k, v in context.items():
                if isinstance(v, str):
                    preamble += f'{k} = r"""{v}"""\n'
                else:
                    preamble += f"{k} = {repr(v)}\n"

        full_code = self._sanitize_code(preamble + "\n" + code)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"    # always write as UTF-8
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",   # always read as UTF-8
            )
            stdout  = result.stdout.strip()
            stderr  = result.stderr.strip()
            success = result.returncode == 0

        except subprocess.TimeoutExpired:
            stdout  = ""
            stderr  = f"Execution timed out after {self.timeout}s"
            success = False

        finally:
            os.unlink(tmp_path)

        output = stdout if success else f"ERROR:\n{stderr}"
        return {"stdout": stdout, "stderr": stderr, "success": success, "output": output}