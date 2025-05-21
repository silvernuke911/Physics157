import numpy as np 
import math 

# Constants and given data
bitflips_per256mb_permonth = 1 
ram_GB = 32 
days = 15 
d_month = 30 # approx

def poisson_probs(k, lambda_val):
    # Possion distribution for probability
    return np.exp(-lambda_val)*(lambda_val**k)/(math.factorial(k))

# convert MB to GB
ram_MB = ram_GB * 1024
# calculate how many 256MB units are in RAM
ram_units = ram_MB/256
# calculate the time proportion
time_prop = days/d_month
# calculate the lambda value
lambda_val = bitflips_per256mb_permonth*ram_units*time_prop
# Calculate probability of 0 bit flips using custom Poisson function
prob_zero_flips = poisson_probs(0, lambda_val)
# Probability of at least 1 bit flip is complement of 0 flips
prob_at_least_one = 1 - prob_zero_flips
# Print results
print(f"Calculated lambda (expected bit flips): {lambda_val}")
print(f"Probability of zero bit flips: {prob_zero_flips:.10f}")
print(f"Probability of at least one bit flip: {prob_at_least_one:.10f}")