# %% Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import io

# Configure settings
warnings.filterwarnings('ignore')
sns.set_style('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# %% 1. Data Acquisition
ticker = 'GOOGL'
# Use a reasonably long period to capture different market regimes
start_date = '2018-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

print(f"Fetching {ticker} data from {start_date} to {end_date}...")
googl_data = yf.download(ticker, start=start_date, end=end_date)
print("Data fetched successfully.")
print(googl_data.tail())

# %% 2. Data Preparation - Calculate Log Returns
# Log returns are generally preferred for financial time series analysis
googl_data['log_return'] = np.log(googl_data['Adj Close'] / googl_data['Adj Close'].shift(1))
# Remove the first NaN value
googl_returns = googl_data['log_return'].dropna()

print(f"\nLog returns calculated. Number of observations: {len(googl_returns)}")
print(googl_returns.describe())

# %% 3. Exploratory Data Analysis (EDA)

# Plot closing price
plt.figure()
googl_data['Adj Close'].plot(title=f'{ticker} Adjusted Closing Price')
plt.ylabel('Price (USD)')
plt.grid(True)
price_plot_path = 'googl_price_plot.png'
plt.savefig(price_plot_path)
plt.close() # Close plot to prevent displaying in console output during script run

# Plot log returns
plt.figure()
googl_returns.plot(title=f'{ticker} Daily Log Returns', alpha=0.7)
plt.ylabel('Log Return')
plt.grid(True)
returns_plot_path = 'googl_returns_plot.png'
plt.savefig(returns_plot_path)
plt.close()

# Histogram of log returns vs Normal distribution
plt.figure()
sns.histplot(googl_returns, kde=True, stat='density', label='Log Returns', bins=50)
# Overlay a normal distribution with the same mean and std dev
mu_sample, std_sample = stats.norm.fit(googl_returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu_sample, std_sample)
plt.plot(x, p, 'k', linewidth=2, label='Normal Fit')
plt.title(f'{ticker} Log Returns Distribution vs Normal')
plt.legend()
hist_plot_path = 'googl_returns_hist_plot.png'
plt.savefig(hist_plot_path)
plt.close()

# QQ Plot
plt.figure()
stats.probplot(googl_returns, dist="norm", plot=plt)
plt.title(f'{ticker} Log Returns QQ Plot vs Normal')
qq_plot_path = 'googl_qq_plot.png'
plt.savefig(qq_plot_path)
plt.close()

# Normality Test (Jarque-Bera)
jb_test = stats.jarque_bera(googl_returns)
print(f"\nJarque-Bera Normality Test Results:")
print(f"  Statistic: {jb_test[0]:.4f}")
print(f"  p-value: {jb_test[1]:.4f}")
if jb_test[1] < 0.05:
    print("  Conclusion: Reject normality (p < 0.05). Returns exhibit non-normal characteristics (likely fat tails/skewness).")
else:
    print("  Conclusion: Cannot reject normality (p >= 0.05).")


# %% 4. Bayesian Model Specification (Student's t)

# Define the model using PyMC
coords = {"observation": googl_returns.index}
with pm.Model(coords=coords) as model_t:
    # --- Priors ---
    # Prior for mean (mu): Normal distribution centered around 0
    mu = pm.Normal('mu', mu=0.0, sigma=0.01) # Centered at 0, small std dev

    # Prior for scale (sigma): Half-Cauchy is weakly informative for scale parameters
    # A small beta implies less informative prior.
    sigma = pm.HalfCauchy('sigma', beta=0.05) # Prior on daily volatility scale

    # Prior for degrees of freedom (nu): Exponential distribution.
    # Lower nu indicates fatter tails. nu > 2 for finite variance.
    # Common prior: Shifted exponential Exp(1/29) + 2 to avoid issues near nu=2
    # and concentrate mass between 2 and ~30, allowing for fat tails.
    # For simplicity here, let's use a simpler Gamma or Exp starting from a reasonable minimum.
    # nu_minus_2 = pm.Exponential('nu_minus_2', 1.0/29.0) # Mean of nu_minus_2 is 29
    # nu = pm.Deterministic('nu', nu_minus_2 + 2) # nu = nu_minus_2 + 2 -> mean nu is 31

    # Alternative simpler prior for nu: Gamma distribution
    # Shape/alpha controls the shape, Rate/beta controls the scale. mean = alpha/beta
    # Let's try a Gamma that allows smaller values but penalizes very small ones.
    # Mean around 5-10? alpha=2, beta=0.1 -> mean=20. alpha=2, beta=0.2 -> mean=10
    nu = pm.Gamma('nu', alpha=2, beta=0.1) # Prior mean nu = 20, allows values down to ~3-4 realistically


    # --- Likelihood ---
    # Student's t-distribution for the observed log returns
    returns_obs = pm.StudentT('returns_obs',
                              nu=nu,
                              mu=mu,
                              sigma=sigma, # Note: PyMC uses sigma (scale), not std dev directly for StudentT
                              observed=googl_returns.values,
                              dims="observation")

# Visualize model graph (optional)
# try:
#     graph = pm.model_to_graphviz(model_t)
#     graph.render('model_graph', format='png', view=False, cleanup=True)
#     model_graph_path = 'model_graph.png'
# except ImportError:
#     print("Graphviz not installed or not found in PATH. Skipping model graph generation.")
#     model_graph_path = None
model_graph_path = None # Simpler not to depend on graphviz installation


# %% 5. MCMC/HMC Sampling
# Use NUTS sampler (an adaptive HMC method)
n_draws = 2000
n_tune = 1500
n_chains = 4

print(f"\nStarting MCMC sampling (NUTS)...")
print(f"Sampler settings: draws={n_draws}, tune={n_tune}, chains={n_chains}")

with model_t:
    idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, target_accept=0.9, random_seed=42)
    # Sample posterior predictive for model checking
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))

print("MCMC sampling complete.")

# %% 6. Convergence Diagnostics
print("\n--- Convergence Diagnostics ---")
summary = az.summary(idata, var_names=['mu', 'sigma', 'nu'])
print(summary)

# Check R-hat and ESS
rhat_ok = (summary['r_hat'] < 1.01).all()
ess_bulk_ok = (summary['ess_bulk'] > 400).all() # Rule of thumb: > 400 for reliable estimates
ess_tail_ok = (summary['ess_tail'] > 400).all()

print(f"\nR-hat values acceptable (< 1.01)? {'Yes' if rhat_ok else 'No'}")
print(f"Bulk Effective Sample Size (ESS) acceptable (> 400)? {'Yes' if ess_bulk_ok else 'No'}")
print(f"Tail Effective Sample Size (ESS) acceptable (> 400)? {'Yes' if ess_tail_ok else 'No'}")

if not (rhat_ok and ess_bulk_ok and ess_tail_ok):
    print("Warning: Convergence issues detected. Results might be unreliable. Consider increasing tune/draws or reparametrization.")

# Plot traces
trace_plot = az.plot_trace(idata, var_names=['mu', 'sigma', 'nu'])
plt.tight_layout()
trace_plot_path = 'mcmc_trace_plots.png'
plt.savefig(trace_plot_path)
plt.close()


# %% 7. Results Analysis - Posterior Distributions
print("\n--- Posterior Distribution Analysis ---")

# Plot posterior distributions
post_plot = az.plot_posterior(idata, var_names=['mu', 'sigma', 'nu'])
plt.tight_layout()
posterior_plot_path = 'mcmc_posterior_plots.png'
plt.savefig(posterior_plot_path)
plt.close()

# Get posterior means for parameters
posterior_mu = idata.posterior['mu'].mean().item()
posterior_sigma = idata.posterior['sigma'].mean().item()
posterior_nu = idata.posterior['nu'].mean().item()

print(f"\nPosterior Mean Estimates:")
print(f"  mu (Mean Daily Log Return): {posterior_mu:.6f}")
print(f"  sigma (Daily Volatility Scale): {posterior_sigma:.6f}")
print(f"  nu (Degrees of Freedom): {posterior_nu:.4f}")

# Interpretation of nu
if posterior_nu < 5:
    tail_desc = "Very Fat Tails (Significant extreme event risk)"
elif posterior_nu < 10:
    tail_desc = "Fat Tails (Higher probability of extreme events than Normal)"
elif posterior_nu < 30:
    tail_desc = "Moderately Fat Tails (Approaching Normal but still heavier)"
else:
    tail_desc = "Near Normal Tails"
print(f"  Tail Risk Interpretation (based on nu={posterior_nu:.2f}): {tail_desc}")

# %% 8. Financial Insights & Metrics

# --- Volatility ---
# Daily volatility (posterior distribution)
daily_vol_posterior = idata.posterior['sigma']

# Annualized volatility (assuming 252 trading days)
annualized_vol_posterior = daily_vol_posterior * np.sqrt(252)
annualized_vol_mean = annualized_vol_posterior.mean().item()
annualized_vol_hdi = az.hdi(annualized_vol_posterior.values.flatten(), hdi_prob=0.94) # 94% HDI is common

print(f"\nAnnualized Volatility:")
print(f"  Mean Estimate: {annualized_vol_mean:.4f} ({annualized_vol_mean*100:.2f}%)")
print(f"  94% Highest Density Interval (HDI): [{annualized_vol_hdi[0]:.4f}, {annualized_vol_hdi[1]:.4f}]")

# Plot Annualized Volatility Posterior
plt.figure()
sns.histplot(annualized_vol_posterior.values.flatten(), kde=True, stat='density', bins=50)
plt.axvline(annualized_vol_mean, color='r', linestyle='--', label=f'Mean: {annualized_vol_mean:.3f}')
plt.axvline(annualized_vol_hdi[0], color='g', linestyle=':', label='94% HDI')
plt.axvline(annualized_vol_hdi[1], color='g', linestyle=':')
plt.title(f'{ticker} Posterior Distribution of Annualized Volatility')
plt.xlabel('Annualized Volatility')
plt.legend()
ann_vol_plot_path = 'googl_annualized_vol_posterior.png'
plt.savefig(ann_vol_plot_path)
plt.close()

# --- Value-at-Risk (VaR) ---
# Calculate VaR using the posterior mean parameters of the fitted t-distribution
# VaR(alpha) is the quantile of the return distribution
alpha_levels = [0.05, 0.01] # 95% and 99% VaR

print("\nValue-at-Risk (VaR) Estimates (using posterior mean parameters):")
for alpha in alpha_levels:
    # Use the ppf (percent point function, inverse of cdf) of the t-distribution
    # Note: Scipy's t takes loc=mu, scale=sigma, df=nu
    var_estimate = stats.t.ppf(alpha, df=posterior_nu, loc=posterior_mu, scale=posterior_sigma)
    print(f"  Daily VaR ({1-alpha:.0%}): {var_estimate:.6f} (Suggests a {abs(var_estimate)*100:.2f}% loss or worse {alpha*100:.0f}% of the time)")

# Note: A more complete Bayesian approach would compute the VaR for each posterior sample
# and get a posterior distribution *for* the VaR itself. For simplicity, we use the mean params.

# --- Posterior Predictive Checks (PPC) ---
# Compare the observed data distribution to distributions simulated from the fitted model
print("\nPerforming Posterior Predictive Checks (PPC)...")
ppc_data = idata.posterior_predictive['returns_obs'] # Shape: (n_chains, n_draws, n_observations)
# Flatten chains and draws, take a subset of simulations for plotting efficiency
ppc_samples_flat = ppc_data.values.reshape(-1, ppc_data.shape[-1])
n_ppc_plots = 50 # Number of simulated datasets to overlay
indices_ppc = np.random.choice(ppc_samples_flat.shape[0], n_ppc_plots, replace=False)

plt.figure()
sns.histplot(googl_returns, kde=False, stat='density', color='black', label='Observed Data', bins=50, alpha=0.6)
for i in indices_ppc:
    sns.histplot(ppc_samples_flat[i, :], kde=False, stat='density', alpha=0.05, color='steelblue', bins=50)
# Add one line with label for legend clarity
sns.histplot(ppc_samples_flat[indices_ppc[0], :], kde=False, stat='density', alpha=0, color='steelblue', label='Posterior Predictive Samples', bins=50)
plt.title(f'{ticker} Posterior Predictive Check: Observed vs Simulated Returns')
plt.xlabel('Log Return')
plt.legend()
ppc_plot_path = 'googl_ppc_plot.png'
plt.savefig(ppc_plot_path)
plt.close()

print("PPC plot generated. Check if simulated distributions visually match the observed data distribution.")

# %% 9. Generate Markdown Report

# Create plot paths dictionary
plot_paths = {
    "price_plot": price_plot_path,
    "returns_plot": returns_plot_path,
    "hist_plot": hist_plot_path,
    "qq_plot": qq_plot_path,
    "trace_plot": trace_plot_path,
    "posterior_plot": posterior_plot_path,
    "ann_vol_plot": ann_vol_plot_path,
    "ppc_plot": ppc_plot_path,
    "model_graph": model_graph_path # Could be None
}

# Use io.StringIO to build the markdown string
md_content = io.StringIO()

md_content.write(f"# Bayesian Financial Analysis of {ticker} Stock\n\n")
md_content.write(f"**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}\n")
md_content.write(f"**Analysis Period:** {start_date} to {end_date}\n\n")
md_content.write("This report analyzes the financial characteristics of GOOGL stock, focusing on daily log returns, using Bayesian inference with MCMC/HMC methods (specifically, the NUTS sampler in PyMC). We model the returns using a Student's t-distribution to capture potential fat tails, providing insights into volatility and tail risk.\n\n")

md_content.write("## 1. Data Overview\n\n")
md_content.write(f"Historical adjusted closing prices for {ticker} were obtained from Yahoo Finance.\n")
md_content.write(f"![{ticker} Price]({plot_paths['price_plot']})\n\n")
md_content.write("Daily log returns were calculated for the analysis.\n")
md_content.write(f"![{ticker} Log Returns]({plot_paths['returns_plot']})\n\n")
md_content.write("Basic statistics of log returns:\n")
md_content.write("```\n")
md_content.write(googl_returns.describe().to_string())
md_content.write("\n```\n\n")

md_content.write("## 2. Exploratory Data Analysis (EDA)\n\n")
md_content.write("We examine the distribution of log returns to assess normality.\n")
md_content.write(f"![{ticker} Log Returns Histogram vs Normal]({plot_paths['hist_plot']})\n\n")
md_content.write(f"![{ticker} Log Returns QQ Plot vs Normal]({plot_paths['qq_plot']})\n\n")
md_content.write(f"**Jarque-Bera Normality Test:**\nStatistic: {jb_test[0]:.4f}, p-value: {jb_test[1]:.4f}\n")
md_content.write(f"*Conclusion:* {'Reject normality' if jb_test[1] < 0.05 else 'Cannot reject normality'}. The histogram and QQ plot suggest deviations from normality, particularly heavier tails, motivating the use of a Student's t-distribution.\n\n")

md_content.write("## 3. Bayesian Model: Student's t-Distribution\n\n")
md_content.write("We model the daily log returns `r` as following a Student's t-distribution:\n")
md_content.write("`r ~ StudentT(ν, μ, σ)`\n\n")
md_content.write("Where:\n")
md_content.write("- `μ`: Mean daily log return.\n")
md_content.write("- `σ`: Scale parameter (related to volatility).\n")
md_content.write("- `ν`: Degrees of freedom (controls tail fatness; lower `ν` means fatter tails).\n\n")
md_content.write("Priors used:\n")
md_content.write("- `μ ~ Normal(0, 0.01)`\n")
md_content.write("- `σ ~ HalfCauchy(0.05)`\n")
md_content.write("- `ν ~ Gamma(α=2, β=0.1)`\n\n")
# if plot_paths['model_graph']:
#     md_content.write("Model structure:\n")
#     md_content.write(f"![Model Graph]({plot_paths['model_graph']})\n\n")

md_content.write("## 4. MCMC Sampling & Convergence\n\n")
md_content.write(f"The model parameters were estimated using the NUTS sampler (HMC variant) with {n_chains} chains, {n_draws} draws each after {n_tune} tuning steps.\n\n")
md_content.write("**Convergence Diagnostics Summary:**\n")
md_content.write("```\n")
md_content.write(summary.to_string())
md_content.write("\n```\n")
md_content.write(f"*Diagnostics Check:* R-hat {'OK' if rhat_ok else 'Issue'}, ESS Bulk {'OK' if ess_bulk_ok else 'Issue'}, ESS Tail {'OK' if ess_tail_ok else 'Issue'}.\n")
md_content.write(f"![MCMC Trace Plots]({plot_paths['trace_plot']})\n*Trace plots show the sampling paths for each parameter across chains. They should look like stationary 'fuzzy caterpillars' indicating good mixing and convergence.*\n\n")

md_content.write("## 5. Posterior Parameter Estimates\n\n")
md_content.write("The posterior distributions represent our updated beliefs about the parameters after observing the data.\n")
md_content.write(f"![Posterior Distributions]({plot_paths['posterior_plot']})\n\n")
md_content.write("**Posterior Mean Estimates:**\n")
md_content.write(f"- `μ` (Mean Daily Log Return): {posterior_mu:.6f}\n")
md_content.write(f"- `σ` (Daily Volatility Scale): {posterior_sigma:.6f}\n")
md_content.write(f"- `ν` (Degrees of Freedom): {posterior_nu:.2f}\n\n")
md_content.write(f"**Tail Risk Interpretation:** The estimated degrees of freedom (`ν` ≈ {posterior_nu:.2f}) is relatively low, indicating **{tail_desc}**. This suggests that extreme positive or negative daily returns are significantly more likely for GOOGL than would be predicted by a Normal distribution.\n\n")

md_content.write("## 6. Financial Insights & Metrics\n\n")
md_content.write("### Volatility\n")
md_content.write("The posterior distribution for `σ` gives us a probabilistic estimate of daily volatility. We can annualize this (multiplying by `sqrt(252)`).\n")
md_content.write(f"![Annualized Volatility Posterior]({plot_paths['ann_vol_plot']})\n")
md_content.write(f"- **Mean Annualized Volatility:** {annualized_vol_mean*100:.2f}%\n")
md_content.write(f"- **94% HDI for Annualized Volatility:** [{annualized_vol_hdi[0]*100:.2f}%, {annualized_vol_hdi[1]*100:.2f}%]\n")
md_content.write("*The HDI provides a credible range for the annualized volatility.*\n\n")

md_content.write("### Value-at-Risk (VaR)\n")
md_content.write("Using the posterior mean parameters of the fitted Student's t-distribution, we estimate VaR.\n")
for alpha in alpha_levels:
    var_estimate = stats.t.ppf(alpha, df=posterior_nu, loc=posterior_mu, scale=posterior_sigma)
    md_content.write(f"- **Daily VaR ({1-alpha:.0%}): {var_estimate:.4f}** (Implies a {abs(var_estimate)*100:.2f}% loss or worse is expected on {alpha*100:.0f}% of trading days, based on this model).\n")
md_content.write("*Note: These VaR estimates incorporate the fat tails captured by the t-distribution.*\n\n")

md_content.write("### Model Fit Assessment (Posterior Predictive Check)\n")
md_content.write("We simulate datasets from the fitted model and compare their distribution to the observed data.\n")
md_content.write(f"![Posterior Predictive Check]({plot_paths['ppc_plot']})\n")
md_content.write("*The plot shows the histogram of observed returns (black) overlaid with histograms from multiple datasets simulated using parameters drawn from the posterior (blue). A good fit is indicated if the simulated data generally resembles the observed data distribution.*\n\n")

md_content.write("## 7. Conclusion & Limitations\n\n")
md_content.write(f"This Bayesian analysis suggests that GOOGL daily log returns over the period {start_date} to {end_date} are well-described by a Student's t-distribution. Key findings include:\n")
md_content.write(f"- **Significant Tail Risk:** The estimated low degrees of freedom (`ν` ≈ {posterior_nu:.2f}) confirms the presence of fat tails, meaning extreme price movements are more probable than under a Normal distribution assumption.\n")
md_content.write(f"- **Volatility Estimate:** The annualized volatility is estimated to be around {annualized_vol_mean*100:.2f}%, with a 94% credible interval of [{annualized_vol_hdi[0]*100:.2f}%, {annualized_vol_hdi[1]*100:.2f}%].\n")
md_content.write(f"- **Risk Metrics:** The calculated VaR values reflect the identified tail risk.\n\n")
md_content.write("**Limitations:**\n")
md_content.write("- **Constant Parameters:** The model assumes constant `μ`, `σ`, and `ν` over the entire analysis period. In reality, volatility (and potentially other parameters) varies over time (volatility clustering). More advanced models like Stochastic Volatility or GARCH models could address this.\n")
md_content.write("- **Model Choice:** While the Student's t-distribution is an improvement over the Normal, other distributions (e.g., skewed t) or mixture models might capture return dynamics even better.\n")
md_content.write("- **Exogenous Factors:** The model does not incorporate external factors (e.g., market indices, macroeconomic news).\n\n")
md_content.write("This analysis provides a quantitative, probabilistic assessment of GOOGL's return characteristics, leveraging the power of Bayesian inference and MCMC/HMC methods.\n")

# Save the markdown content to a file
md_filename = 'GOOGL_Bayesian_Analysis_Report.md'
with open(md_filename, 'w') as f:
    f.write(md_content.getvalue())

print(f"\nAnalysis complete. Report saved as '{md_filename}'")
print("Generated plot files:")
for key, path in plot_paths.items():
    if path and os.path.exists(path):
        print(f"- {path}")
    # elif key == 'model_graph' and not path:
    #     print("- model_graph.png (skipped)")

# Clean up StringIO object
md_content.close()

# End of script
# %%