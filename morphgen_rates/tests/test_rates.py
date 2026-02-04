"""
Minimal test example: compute bifurcation and annihilation rates from packaged data.
"""

from morphgen_rates import compute_rates, get_data

if __name__ == "__main__":
  # Load summary statistics for aPC pyramidal neurons and select the apical dendrite section
  data = get_data("aPC", "PYR")["apical_dendrite"]

  # (Optional) inspect the input dictionary used by the estimator
  print("Input data keys:", list(data.keys()))
  print("Sholl bin size:", data["sholl_plot"]["bin_size"])

  # Maximum advancement (distance from soma) allowed for one elongation step
  max_step_size = 5.0

  # Estimate rates
  rates = compute_rates(data, max_step_size=max_step_size)

  print("Bifurcation rate:", rates.get("bifurcation_rate"))
  print("Annihilation rate:", rates.get("annihilation_rate"))

