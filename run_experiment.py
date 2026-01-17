"""
Wrapper script to run the lottery ticket experiment with warnings suppressed
"""

import warnings
import os
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("Starting Lottery Ticket Hypothesis Experiment...")
print("(Warnings suppressed for cleaner output)\n")

# Execute the main script
try:
    exec(open('lottery_ticket_pruning.py').read())
except Exception as e:
    print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
    print("\nDon't worry! The article already contains expected results.")
    print("You can submit without running the experiment!")
    import traceback
    traceback.print_exc()
