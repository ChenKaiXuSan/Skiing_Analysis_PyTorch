#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, '/workspace/code/angle')

from main import (
    ID_TO_INDEX, process_person
)

# Test with run_3 only
input_path = Path("/workspace/data/fused_smoothed_results/run_3.npy")
output_dir = Path("/workspace/data/angle_outputs/run_3")

print(f"Processing run_3...")
try:
    process_person(input_path, output_dir)
    print("Done!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Check output
print(f"\nChecking output files:")
files = sorted(output_dir.glob("*.csv"))
for f in files:
    print(f"  ✓ {f.name}")
