import sys
from pathlib import Path

# The tests share problem loaders/validators with the benchmark scripts at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
