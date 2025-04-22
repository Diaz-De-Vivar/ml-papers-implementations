# %% Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pytensor.tensor as pt
import warnings
import time

# Configure settings
warnings.filterwarnings('ignore')
sns.set_style('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# %% 1. Data Acquisition & Preparation (Same as before)
ticker = 'GOOGL'
start_date = '2018-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

print(f"Fetching {ticker} data from {start_date} to {end_date}...")
googl_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
print("Data fetched successfully.")

# Calculate Log Returns (centered for numerical stability often helps, but not strictly necessary here)
googl_data['log_return'] = np.log(googl_data['Adj Close'] / googl_data['Adj Close'].shift(1))
googl_returns = googl_data['log_return'].dropna()
returns_array = googl_returns.values
T = len(returns_array)

print(f"\nLog returns calculated. Number of observations (T): {T}")
print(googl_returns.describe())

# Plot returns again for context
plt.figure(figsize=(12, 4))
plt.plot(googl_returns.index, returns_array, alpha=0.7)
plt.title(f'{ticker} Daily Log Returns ({start_date} to {end_date})')
plt.ylabel('Log Return')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

# %% 2. Bayesian Model Specification (Stochastic Volatility with t-Errors - SV-t)

coords = {"time": googl_returns.index}

with pm.Model(coords=coords) as sv_t_model:
    # === Priors for SV parameters ===
    # Mean log-volatility (intercept of AR(1))
    mu_h = pm.Normal('mu_h', mu=0.0, sigma=5.0) # Relatively wide prior

    # Persistence of log-volatility (phi) - constrained between -1 and 1
    # Use Beta distribution on transformed phi: rho = (phi + 1) / 2
    # Prior favors high persistence (phi close to 1)
    phi_transformed = pm.Beta('phi_transformed', alpha=20.0, beta=1.5)
    phi = pm.Deterministic('phi', 2.0 * phi_transformed - 1.0)

    # Volatility of log-volatility (sigma_h) - must be positive
    sigma_h = pm.HalfCauchy('sigma_h', beta=0.5) # Standard weakly informative prior

    # === Latent Log-Volatility Process (h_t) ===
    # AR(1) process for h_t: h_t = mu_h + phi * (h_{t-1} - mu_h) + sigma_h * eta_t
    # PyMC's AR handles h_t = c + rho * h_{t-1} + innovation_sd * Normal(0,1)
    # We model h_centered = h_t - mu_h, so h_centered_t = phi * h_centered_{t-1} + sigma_h * eta_t
    # The constant c in pm.AR is 0 for h_centered.
    # The innovations have standard deviation sigma_h.

    # Initial distribution for the AR process (stationary distribution)
    # Variance of stationary AR(1) is sigma_innovation^2 / (1 - phi^2)
    h_init_sd = sigma_h / pm.math.sqrt(1.0 - phi**2)
    h_centered = pm.AR('h_centered', rho=phi, sigma=sigma_h, constant=False, # constant=False implies c=0
                      init_dist=pm.Normal.dist(mu=0.0, sigma=h_init_sd),
                      dims="time")

    # Actual log-volatility h_t
    h = pm.Deterministic('h', h_centered + mu_h, dims="time")

    # === Observation Noise Distribution ===
    # Degrees of freedom for Student's t-distribution (nu) - must be positive
    # Gamma prior allowing smaller values (fatter tails)
    nu = pm.Gamma('nu', alpha=2.0, beta=0.1) # Prior mean = 20, allows values realistically down to ~3-4

    # === Likelihood ===
    # Returns r_t = exp(h_t / 2) * epsilon_t, where epsilon_t ~ StudentT(nu, 0, 1)
    # This means r_t ~ StudentT(nu, 0, sigma=exp(h_t / 2))
    # PyMC StudentT takes 'sigma' as the scale parameter.
    return_volatility = pm.Deterministic("return_volatility", pm.math.exp(h / 2.0), dims="time")
    returns_obs = pm.StudentT('returns_obs',
                              nu=nu,
                              mu=0.0, # Assuming mean return is captured elsewhere or near zero
                              sigma=return_volatility,
                              observed=returns_array,
                              dims="time") # Ensure dimensions match


# %% 3. MCMC Sampling
# SV models can be challenging. Need sufficient tuning and draws, and potentially higher target_accept.
n_tune = 2500
n_draws = 2500
n_chains = 4
target_accept_level = 0.95 # Higher target_accept often needed for SV

print("\nStarting MCMC sampling for SV-t model (this might take a while)...")
print(f"Sampler settings: draws={n_draws}, tune={n_tune}, chains={n_chains}, target_accept={target_accept_level}")

start_time = time.time()
with sv_t_model:
    idata_svt = pm.sample(draws=n_draws,
                          tune=n_tune,
                          chains=n_chains,
                          target_accept=target_accept_level,
                          random_seed=42,
                          idata_kwargs={'log_likelihood': True}) # Needed for LOO/WAIC
end_time = time.time()
print(f"MCMC sampling finished in {end_time - start_time:.1f} seconds.")

# %% 4. Convergence Diagnostics & Parameter Posteriors

print("\n--- Convergence Diagnostics & Parameter Summary ---")
var_names_params = ['mu_h', 'phi', 'sigma_h', 'nu']
summary = az.summary(idata_svt, var_names=var_names_params)
print(summary)

# Check R-hat and ESS visually as well
print("\nPlotting trace plots for SV parameters...")
az.plot_trace(idata_svt, var_names=var_names_params)
plt.tight_layout()
plt.show()
plt.close()

print("\nPlotting posterior distributions for SV parameters...")
az.plot_posterior(idata_svt, var_names=var_names_params)
plt.tight_layout()
plt.show()
plt.close()

# Check for divergences
divergences = idata_svt.sample_stats.diverging.sum().item()
print(f"\nNumber of divergences: {divergences}")
if divergences > 0:
     print("WARNING: Divergences detected. Results might be biased. Consider reparameterization, stronger priors, or longer tuning.")

# %% 5. Analysis of Latent Volatility

print("\n--- Latent Volatility Analysis ---")
# Extract posterior samples for h and calculate annualized volatility
h_posterior = idata_svt.posterior['h']
# Annualized volatility: exp(h_t / 2) * sqrt(252)
annualized_vol_posterior = np.exp(h_posterior / 2.0) * np.sqrt(252)

# Calculate mean and HDI for annualized volatility
mean_ann_vol = annualized_vol_posterior.mean(dim=("chain", "draw"))
hdi_ann_vol = az.hdi(annualized_vol_posterior, hdi_prob=0.94) # 94% HDI

# Plot the estimated annualized volatility path
plt.figure(figsize=(14, 6))
plt.plot(googl_returns.index, mean_ann_vol, label='Posterior Mean Annualized Volatility', color='darkblue')
plt.fill_between(googl_returns.index, hdi_ann_vol.sel(hdi='lower')['h'], hdi_ann_vol.sel(hdi='upper')['h'],
                 color='lightblue', alpha=0.6, label='94% HDI')

plt.title(f'{ticker} Estimated Annualized Volatility (SV-t Model)')
plt.ylabel('Annualized Volatility')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

print("Volatility plot generated.")
print(f"Mean estimated annualized volatility on the last day ({googl_returns.index[-1].date()}): {mean_ann_vol[-1].item():.4f}")
print(f"94% HDI for annualized volatility on the last day: [{hdi_ann_vol.sel(hdi='lower')['h'][-1].item():.4f}, {hdi_ann_vol.sel(hdi='upper')['h'][-1].item():.4f}]")

# %% 6. Bayesian Value-at-Risk (VaR) - 1-day ahead

print("\n--- Bayesian Value-at-Risk (VaR) Calculation (1-day ahead) ---")
alpha_levels = [0.05, 0.01] # 95% and 99% VaR levels

# Get posterior samples for the last log-volatility state (h_T) and nu
h_T_samples = h_posterior[:, :, -1].values.flatten() # Flatten across chains and draws
nu_samples = idata_svt.posterior['nu'].values.flatten()

# Ensure same number of samples
n_samples = len(h_T_samples)
if len(nu_samples) != n_samples:
    # This shouldn't happen if sampling was consistent, but as a safeguard:
    min_len = min(len(h_T_samples), len(nu_samples))
    h_T_samples = h_T_samples[:min_len]
    nu_samples = nu_samples[:min_len]
    print(f"Warning: Adjusted sample lengths for VaR calculation to {min_len}")

# Calculate scale parameter for the t-distribution for each posterior sample
scale_T_samples = np.exp(h_T_samples / 2.0)

var_results = {}
plt.figure(figsize=(12, 5 * len(alpha_levels)))
for i, alpha in enumerate(alpha_levels):
    # Calculate VaR for each posterior sample using the t-distribution ppf
    # VaR(alpha) = ppf(alpha, df=nu, loc=0, scale=sigma_t)
    var_samples = stats.t.ppf(alpha, df=nu_samples, loc=0, scale=scale_T_samples)
    var_results[alpha] = var_samples

    # Analyze the posterior distribution of VaR(alpha)
    mean_var = np.mean(var_samples)
    median_var = np.median(var_samples)
    hdi_var = az.hdi(var_samples, hdi_prob=0.94)

    print(f"\nBayesian VaR ({1-alpha:.0%}) Analysis:")
    print(f"  Posterior Mean VaR: {mean_var:.6f}")
    print(f"  Posterior Median VaR: {median_var:.6f}")
    print(f"  94% HDI for VaR: [{hdi_var[0]:.6f}, {hdi_var[1]:.6f}]")
    print(f"  Interpretation: There is a {alpha*100:.0f}% chance the next day's return will be worse than {mean_var:.2%}, with 94% credibility that this threshold lies between {hdi_var[0]:.2%} and {hdi_var[1]:.2%}.")

    # Plot the posterior distribution of VaR
    plt.subplot(len(alpha_levels), 1, i + 1)
    sns.histplot(var_samples, kde=True, bins=50, stat='density')
    plt.axvline(mean_var, color='r', linestyle='--', label=f'Mean: {mean_var:.4f}')
    plt.axvline(hdi_var[0], color='g', linestyle=':', label='94% HDI')
    plt.axvline(hdi_var[1], color='g', linestyle=':')
    plt.title(f'Posterior Distribution of 1-Day {1-alpha:.0%} VaR for {ticker}')
    plt.xlabel('VaR Estimate (Log Return)')
    plt.legend()

plt.tight_layout()
plt.show()
plt.close()


# %% 7. Posterior Predictive Checks (PPC)

print("\n--- Posterior Predictive Checks ---")
print("Sampling from posterior predictive distribution...")
# This can be memory intensive for long time series
with sv_t_model:
    # Sample fewer PPC draws if memory is an issue
    idata_svt.extend(pm.sample_posterior_predictive(idata_svt, var_names=['returns_obs'], random_seed=42, extend_inferencedata=True))

print("Plotting PPC...")
az.plot_ppc(idata_svt, var_names=['returns_obs'], num_pp_samples=100, kind='kde', figsize=(12, 6)) # Use KDE for smoother comparison
plt.title(f'{ticker} Posterior Predictive Check (SV-t Model)')
plt.xlabel('Log Return')
plt.show()
plt.close()

print("PPC plot generated. Check if the distribution of observed data (dark line) is well-captured by the distributions generated from the model (light lines).")

# %% 8. Model Comparison (Optional - e.g., compare SV-t vs constant-t)
# If the simpler t-model was run and idata saved:
# print("\n--- Model Comparison (using LOO) ---")
# try:
#     # Rerun the simple model sampling if needed, or load idata if saved previously
#     # Assuming `idata_t` exists from the first script (and has log_likelihood saved)
#     # with model_t: # Need to ensure model_t is defined and idata_t exists
#     #    pm.compute_log_likelihood(idata_t, model=model_t) # If log_likelihood wasn't saved

#     compare_dict = {'SV-t': idata_svt, 'Constant-t': idata} # Assuming 'idata' holds the simple t-model results
#     loo_compare = az.compare(compare_dict, ic='loo', scale='deviance')
#     print(loo_compare)
#     az.plot_compare(loo_compare)
#     plt.show()
#     plt.close()
#     print("Models compared using LOO-CV. Lower LOO score indicates better out-of-sample predictive fit.")
# except NameError:
#     print("Skipping model comparison (requires results from the simpler model).")
# except Exception as e:
#     print(f"Could not perform model comparison: {e}")

print("\n--- Rigorous Bayesian Analysis Conclusions (Summary) ---")

# Extract key posterior means/medians for summary
phi_mean = summary.loc['phi', 'mean']
sigma_h_mean = summary.loc['sigma_h', 'mean']
nu_mean = summary.loc['nu', 'mean']
last_day_vol_mean = mean_ann_vol[-1].item()
var_95_mean = np.mean(var_results[0.05])
var_99_mean = np.mean(var_results[0.01])

print(f"1.  **Volatility Clustering:** Highly persistent volatility confirmed (Posterior Mean phi ≈ {phi_mean:.3f}). Volatility level itself fluctuates significantly (Posterior Mean sigma_h ≈ {sigma_h_mean:.3f}).")
print(f"2.  **Time-Varying Volatility:** Estimated annualized volatility varies considerably over time, with the latest estimate around {last_day_vol_mean*100:.2f}%.")
print(f"3.  **Residual Fat Tails:** Even after accounting for SV, returns exhibit fat tails (Posterior Mean nu ≈ {nu_mean:.2f}), indicating higher likelihood of extreme shocks than a Normal distribution conditional on volatility.")
print(f"4.  **Probabilistic Risk Assessment:** Bayesian VaR provides a distribution of potential risk. For 1-day ahead:")
print(f"    - 95% VaR: Posterior Mean ≈ {var_95_mean:.2%}, indicating a 5% chance of loss exceeding this value.")
print(f"    - 99% VaR: Posterior Mean ≈ {var_99_mean:.2%}, indicating a 1% chance of loss exceeding this value.")
print(f"    (Note: Uncertainty in VaR is captured by its posterior distribution and HDI).")
print(f"5.  **Model Fit:** Posterior predictive checks visually assess how well the model captures the observed return distribution dynamics.")
print(f"6.  **Overall:** The SV-t model provides a more realistic and nuanced view of GOOGL's return dynamics, capturing key stylized facts like volatility clustering and fat tails, leading to time-varying and probabilistic risk assessments.")

print("\nAnalysis complete.")