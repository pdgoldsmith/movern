"""python -m movern.ui  →  launch Streamlit app."""
import subprocess
import sys
from pathlib import Path

app = Path(__file__).parent / "app.py"
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:],
    check=True,
)
