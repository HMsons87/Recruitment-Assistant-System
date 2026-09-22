from __future__ import annotations

from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_python_files_parse():
    for path in [ROOT / "app.py", ROOT / "api_client.py", ROOT / "components.py"]:
        ast.parse(path.read_text(encoding="utf-8"))
