"""
Minimal test example: empirical distribution of primary dendrites.

This script loads summary statistics for aPC pyramidal neurons (apical dendrite),
extracts the primary dendrite stats (Count0), and converts them into a discrete
probability distribution using `compute_init_number_probs`.

`probs[i]` is the probability of generating i primary dendrites.
"""

from morphgen_rates import get_data, compute_init_number_probs

if __name__ == "__main__":
  # Load summary statistics and select the apical dendrite section
  data = get_data("aPC", "PYR")["apical_dendrite"]

  # Primary dendrite stats (derived from Count0)
  stats = data["primary_count"]

  mean_primary = float(stats["mean"])
  sd_primary   = float(stats["std"])
  min_primary  = int(stats["min"])
  max_primary  = int(stats["max"])

  probs = compute_init_number_probs(
    mean_primary_dendrites=mean_primary,
    sd_primary_dendrites=sd_primary,
    min_primary_dendrites=min_primary,
    max_primary_dendrites=max_primary,
  )

  print("Primary dendrite stats:")
  print(f"  mean={mean_primary}, std={sd_primary}, min={min_primary}, max={max_primary}")

  print("\nP(# primary dendrites = i):")
  for i, p in enumerate(probs):
    if p > 0:
      print(f"  i={i}: {p:.6f}")
