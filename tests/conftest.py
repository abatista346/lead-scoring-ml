import sys
from pathlib import Path

# Make the pipeline scripts in src/ importable as plain modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
