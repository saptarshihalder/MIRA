"""Regenerate reports from a checkout without an editable installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mira.reporting import cli  # noqa: E402

if __name__ == "__main__":
    cli()
