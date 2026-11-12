import sys
from pathlib import Path

# The tests share problem loaders/validators with benchmark/benchmark_constrained.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
