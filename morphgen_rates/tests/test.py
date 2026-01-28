import pandas as pd
from morphgen_rates import compute_rates, get_data

# Bundle inputs exactly as loaded (no preprocessing)
data = get_data('aPC/PYR/apical')

print(data)

max_step_size = 5.

rates = compute_rates(data, max_step_size=max_step_size)

print("Bifurcation rate:", rates.get("bifurcation_rate"))
print("Annihilation rate:", rates.get("annihilation_rate"))
