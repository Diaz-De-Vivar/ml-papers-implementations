import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Use seaborn for reliable dataset loading
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load Data (Old Faithful using Seaborn)
# Make sure you have seaborn installed: pip install seaborn
try:
    # Use seaborn's built-in geyser dataset
    data = sns.load_dataset('geyser')
    print("Columns in loaded data:", data.columns) # Check column names

    # Standardize the relevant columns (adjust names if needed based on print output)
    # Seaborn 'geyser' dataset typically uses 'duration' and 'waiting'
    eruptions_col = 'duration'
    waiting_col = 'waiting'

    data[f'{eruptions_col}_std'] = (data[eruptions_col] - data[eruptions_col].mean()) / data[eruptions_col].std()
    data[f'{waiting_col}_std'] = (data[waiting_col] - data[waiting_col].mean()) / data[waiting_col].std()

    eruptions = data[f'{eruptions_col}_std'].values
    waiting = data[f'{waiting_col}_std'].values
    print("Data loaded successfully using Seaborn.")
    print(f"Using standardized '{eruptions_col}' for eruptions and '{waiting_col}' for waiting.")

except ImportError:
    print("Seaborn library not found. Please install it: pip install seaborn")
    print("Falling back to dummy data for demonstration.")
    # Fallback dummy data generation (same as before)
    eruptions = np.random.rand(100) * 3 + 1.5
    waiting = 50 + 10 * eruptions + np.random.randn(100) * 5
    eruptions = (eruptions - eruptions.mean()) / eruptions.std()
    waiting = (waiting - waiting.mean()) / waiting.std()
except Exception as e:
    print(f"Failed to load data using Seaborn: {e}")
    print("Falling back to dummy data for demonstration.")
    # Fallback dummy data generation (same as before)
    eruptions = np.random.rand(100) * 3 + 1.5
    waiting = 50 + 10 * eruptions + np.random.randn(100) * 5
    eruptions = (eruptions - eruptions.mean()) / eruptions.std()
    waiting = (waiting - waiting.mean()) / waiting.std()

n_samples_data = len(eruptions) # Store number of data points

# --- Rest of the code (Log Posterior, Gradients, MCMC Functions, Plotting) ---
# --- follows here exactly as in the previous NumPy version response ---
# Make sure the variables 'eruptions' and 'waiting' are correctly used throughout.

# [Insert the definitions for log_likelihood_np, log_prior_np, log_posterior_np,
#  manual_grad_log_posterior, plot_results, metropolis_hastings, gibbs_sampler,
#  slice_sampler_1d, slice_sampler, leapfrog_np, hamiltonian_monte_carlo_np
#  from the previous response here]

# --- Run MH ---
# [Insert the MH running code here]

# --- Run Gibbs ---
# [Insert the Gibbs running code here]

# --- Run Slice Sampler ---
# [Insert the Slice Sampler running code here]

# --- Run HMC (NumPy version) ---
# [Insert the HMC (NumPy) running code here]

# --- NUTS Explanation ---
# [Insert the NUTS explanation here]