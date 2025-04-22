# %% Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.notebook import tqdm  # Use standard tqdm if not in notebook
import warnings
import time
import math

# Configure settings
warnings.filterwarnings('ignore')
sns.set_style('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100
torch.set_default_dtype(torch.float64) # Use float64 for numerical stability

print(f"PyTorch version: {torch.__version__}")

# %% 1. Data Acquisition & Preparation
ticker = 'GOOGL'
start_date = '2018-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

print(f"Fetching {ticker} data from {start_date} to {end_date}...")
googl_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
print("Data fetched successfully.")

# Calculate Log Returns and convert to PyTorch tensor
googl_data['log_return'] = np.log(googl_data['Adj Close'] / googl_data['Adj Close'].shift(1))
googl_returns = googl_data['log_return'].dropna()
returns_tensor = torch.tensor(googl_returns.values, dtype=torch.float64)
T = len(returns_tensor)
time_index = googl_returns.index

print(f"\nLog returns calculated. Number of observations (T): {T}")

# %% 2. Helper Functions for Transformations and Log PDFs

# --- Parameter Transformations ---
# Need to sample parameters in unconstrained space for HMC
# phi: (-1, 1) -> R using atanh or scaled logistic
# sigma_h: (0, inf) -> R using log
# nu: (0, inf) -> R using log (or offset log if nu > min_val)

def transform_phi(phi_unc):
    # Transforms phi_unc from R to phi in (-1, 1)
    return torch.tanh(phi_unc) # Simple tanh transform

def inv_transform_phi(phi):
    # Transforms phi from (-1, 1) to phi_unc in R
    return torch.atanh(phi.clamp(-0.9999, 0.9999)) # Clamp for stability

def transform_sigma_h(log_sigma_h):
    # log_sigma_h (R) to sigma_h (0, inf)
    return torch.exp(log_sigma_h)

def inv_transform_sigma_h(sigma_h):
    # sigma_h (0, inf) to log_sigma_h (R)
    return torch.log(sigma_h.clamp(min=1e-10)) # Clamp for stability

def transform_nu(log_nu):
    # log_nu (R) to nu (0, inf) - potentially shifted if nu must be > 2, e.g.
    # return torch.exp(log_nu) + 2.0
    return torch.exp(log_nu) # Simpler: nu > 0

def inv_transform_nu(nu):
    # nu (0, inf) to log_nu (R)
    # return torch.log((nu - 2.0).clamp(min=1e-10))
    return torch.log(nu.clamp(min=1e-10))

# --- Log Probability Density Functions (using PyTorch for gradients) ---

def log_normal_pdf(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma)**2 - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(math.pi))

def log_student_t_pdf(x, nu, mu, sigma):
    # PyTorch doesn't have a native T distribution PDF easily usable with autograd parameters?
    # Implementing manually. Ref: Wikipedia Student's t-distribution PDF
    # Note: sigma is the scale parameter here
    gam_nu_half = torch.lgamma(nu / 2.0)
    gam_nu_p1_half = torch.lgamma((nu + 1.0) / 2.0)
    log_const = gam_nu_p1_half - gam_nu_half - 0.5 * torch.log(nu * math.pi) - torch.log(sigma)
    log_kernel = -((nu + 1.0) / 2.0) * torch.log(1.0 + ((x - mu) / sigma)**2 / nu)
    return log_const + log_kernel

def log_beta_pdf(x, alpha, beta):
    # For prior on transformed phi = (phi+1)/2
    log_beta_func = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return (alpha - 1.0) * torch.log(x.clamp(min=1e-9)) + \
           (beta - 1.0) * torch.log((1.0 - x).clamp(min=1e-9)) - log_beta_func

def log_halfcauchy_pdf(x, beta):
    # Proportional to: (1 + (x/beta)^2)^-1 for x > 0
    # Log PDF = -log(pi/2) - log(beta) - log(1 + (x/beta)^2) -- ignoring constant doesn't matter for HMC relative probs
    return -torch.log(beta) - torch.log(1.0 + (x / beta)**2) # Ignoring constant

def log_gamma_pdf(x, alpha, beta):
    # alpha=shape, beta=rate
    return alpha * torch.log(beta) - torch.lgamma(alpha) + (alpha - 1.0) * torch.log(x) - beta * x


# %% 3. Define the Log-Posterior Function

def calculate_log_posterior(params_unc, h_latent, returns):
    """
    Calculates the log posterior density: log p(params, h | returns)
    Operates on unconstrained parameters and latent h states.
    Uses PyTorch for autograd.
    """
    T = len(returns)

    # Unpack and transform parameters
    mu_h_unc = params_unc[0]
    phi_unc = params_unc[1]
    log_sigma_h = params_unc[2]
    log_nu = params_unc[3]

    # Transform to constrained space
    mu_h = mu_h_unc # mu_h is already unconstrained
    phi = transform_phi(phi_unc)
    sigma_h = transform_sigma_h(log_sigma_h)
    nu = transform_nu(log_nu)

    # --- Log Priors ---
    # log p(mu_h): Normal(0, 5)
    log_prior_mu_h = log_normal_pdf(mu_h, torch.tensor(0.0), torch.tensor(5.0))

    # log p(phi): Transformed Beta prior on (phi+1)/2 ~ Beta(20, 1.5)
    # Need Jacobian for phi = tanh(phi_unc): d(phi)/d(phi_unc) = sech^2(phi_unc) = 1 - tanh^2(phi_unc) = 1 - phi^2
    phi_transformed_beta = (phi + 1.0) / 2.0
    log_prior_phi_beta = log_beta_pdf(phi_transformed_beta, torch.tensor(20.0), torch.tensor(1.5))
    log_jacobian_phi = torch.log(1.0 - phi**2 + 1e-10) # log |d(phi)/d(phi_unc)|
    log_prior_phi = log_prior_phi_beta + log_jacobian_phi

    # log p(sigma_h): HalfCauchy(0.5) on sigma_h > 0
    # Need Jacobian for sigma_h = exp(log_sigma_h): d(sigma_h)/d(log_sigma_h) = exp(log_sigma_h) = sigma_h
    log_prior_sigma_h_hc = log_halfcauchy_pdf(sigma_h, torch.tensor(0.5))
    log_jacobian_sigma_h = log_sigma_h # log |d(sigma_h)/d(log_sigma_h)| = log(sigma_h)
    log_prior_sigma_h = log_prior_sigma_h_hc + log_jacobian_sigma_h

    # log p(nu): Gamma(2, 0.1) on nu > 0 (or shifted)
    # Need Jacobian for nu = exp(log_nu): d(nu)/d(log_nu) = exp(log_nu) = nu
    log_prior_nu_gamma = log_gamma_pdf(nu, torch.tensor(2.0), torch.tensor(0.1))
    log_jacobian_nu = log_nu # log |d(nu)/d(log_nu)| = log(nu)
    log_prior_nu = log_prior_nu_gamma + log_jacobian_nu

    # Sum scalar priors
    total_log_prior = log_prior_mu_h + log_prior_phi + log_prior_sigma_h + log_prior_nu

    # --- Log Likelihood AR(1) for h ---
    # h_t = mu_h + phi * (h_{t-1} - mu_h) + sigma_h * eta_t
    # Stationary distribution for h_1: Normal(mu_h, sigma_h / sqrt(1 - phi^2))
    h_init_std = (sigma_h / torch.sqrt(1.0 - phi**2 + 1e-10)).clamp(min=1e-10)
    log_lik_h = log_normal_pdf(h_latent[0], mu_h, h_init_std)
    # Transitions for t = 2 to T
    mu_h_t = mu_h + phi * (h_latent[:-1] - mu_h)
    log_lik_h += log_normal_pdf(h_latent[1:], mu_h_t, sigma_h).sum()

    # --- Log Likelihood Returns ---
    # r_t ~ StudentT(nu, 0, sigma=exp(h_t / 2))
    return_scale = torch.exp(h_latent / 2.0).clamp(min=1e-10)
    log_lik_returns = log_student_t_pdf(returns, nu, torch.tensor(0.0), return_scale).sum()

    # --- Total Log Posterior ---
    log_posterior = total_log_prior + log_lik_h + log_lik_returns

    # Check for NaN/Inf (important for stability)
    if torch.isnan(log_posterior) or torch.isinf(log_posterior):
        return torch.tensor(-torch.inf) # Return -inf if calculation fails

    return log_posterior


# %% 4. Implement HMC Step

def hmc_step(current_q, current_log_prob_grad_fn, epsilon, L):
    """ Performs a single HMC step """
    q = current_q.clone().detach().requires_grad_(True)

    # Calculate current potential energy and gradient
    current_U = -current_log_prob_grad_fn(q)
    if torch.isnan(current_U) or torch.isinf(current_U):
       print("Warning: Initial potential energy is NaN/Inf. Rejecting step.")
       return current_q, torch.tensor(-torch.inf), torch.tensor(0.0) # Indicate rejection
    current_U.backward()
    grad_U = q.grad.clone().detach()
    q.grad.zero_() # Clear gradients for next use

    # Sample initial momentum
    p = torch.randn_like(q)
    current_K = 0.5 * torch.sum(p**2)

    # --- Leapfrog Integration ---
    q_new = q.clone().detach().requires_grad_(True)
    p_new = p.clone().detach()

    # Half step for momentum
    p_new -= 0.5 * epsilon * grad_U

    # Full steps for position and momentum
    for _ in range(L):
        # Position update
        q_new.data += epsilon * p_new
        # Momentum half step
        q_new_nograd = q_new.detach().requires_grad_(True) # Avoid graph explosion
        U_new_step = -current_log_prob_grad_fn(q_new_nograd)
        if torch.isnan(U_new_step) or torch.isinf(U_new_step):
             # print(f"Warning: Potential energy NaN/Inf during leapfrog step {_ + 1}. Stopping leapfrog.")
             # Treat as hitting infinite wall - potential rejection later
             U_new = torch.tensor(torch.inf) # Mark as invalid leapfrog
             break # Exit leapfrog early

        U_new_step.backward()
        grad_U_new = q_new_nograd.grad.clone().detach()
        q_new_nograd.grad.zero_()

        p_new -= epsilon * grad_U_new # Complete momentum step if not the last step

    # If loop finished normally, calculate final potential energy
    if not torch.isinf(U_new_step):
      U_new = U_new_step.detach()

    # No need for final half momentum step as K only depends on p_new before this

    # Calculate final kinetic energy
    proposed_K = 0.5 * torch.sum(p_new**2)

    # --- Metropolis-Hastings Acceptance ---
    # Check if leapfrog was valid
    if torch.isinf(U_new) or torch.isnan(U_new):
        accept_prob = 0.0 # Automatically reject if leapfrog hit NaN/Inf
    else:
        # Calculate change in Hamiltonian
        current_H = current_U.detach() + current_K
        proposed_H = U_new + proposed_K
        delta_H = current_H - proposed_H
        accept_prob = torch.min(torch.tensor(1.0), torch.exp(delta_H)).item()

    # Accept or reject
    if np.random.rand() < accept_prob:
        q_next = q_new.detach() # Accept proposal
        log_prob_next = -U_new
    else:
        q_next = current_q.detach() # Reject proposal, stay at current state
        log_prob_next = -current_U.detach()

    return q_next, log_prob_next, accept_prob


# %% 5. Sampling Loop

# --- HMC Settings (NEED TUNING) ---
# These are critical and dataset/model dependent!
num_samples = 5000
burn_in = 2500
epsilon = 0.001 # Step size (might need adjustment, especially for h)
L = 10      # Number of leapfrog steps

# --- Initialization ---
# Parameters (unconstrained)
# Crude initial guesses - should ideally be more informed
init_mu_h_unc = torch.tensor(0.0)
init_phi = torch.tensor(0.95)
init_phi_unc = inv_transform_phi(init_phi)
init_sigma_h = torch.tensor(0.2)
init_log_sigma_h = inv_transform_sigma_h(init_sigma_h)
init_nu = torch.tensor(10.0)
init_log_nu = inv_transform_nu(init_nu)

# Latent states (h) - initialize near prior mean or smooth returns
# init_h = torch.full((T,), init_mu_h_unc.item()) # Simple init
init_h = torch.log(returns_tensor.abs().clamp(min=1e-5)**2) # Rough estimate from returns

# Combine initial state vector q = [params_unc, h_latent]
current_q = torch.cat([
    torch.stack([init_mu_h_unc, init_phi_unc, init_log_sigma_h, init_log_nu]),
    init_h
]).detach()

# --- Storage ---
num_params = 4
samples_q = torch.zeros((num_samples, len(current_q)))
log_probs = torch.zeros(num_samples)
acceptance_rates = []

# --- Log Posterior Function Wrapper (for HMC step) ---
def log_prob_grad_fn(q_vec):
    params_unc = q_vec[:num_params]
    h_latent = q_vec[num_params:]
    return calculate_log_posterior(params_unc, h_latent, returns_tensor)

# --- Run Sampler ---
print("Starting HMC sampling (NumPy/PyTorch)...")
start_time_np = time.time()

# Initial log prob calculation
current_log_prob = log_prob_grad_fn(current_q.requires_grad_(True))
print(f"Initial log probability: {current_log_prob.item()}")

# Burn-in phase
print("Running burn-in...")
accepted_count_burn = 0
for i in tqdm(range(burn_in)):
    current_q, current_log_prob, accept_prob = hmc_step(current_q, log_prob_grad_fn, epsilon, L)
    if i > 0: # Avoid division by zero on first iter
        accepted_count_burn += (accept_prob > 0) # Count accepted steps (approximate)
        if (i + 1) % 100 == 0:
            acc_rate_burn = accepted_count_burn / (i + 1)
            # --- Basic Epsilon Tuning (Very Crude - NUTS does this properly) ---
            # if acc_rate_burn < 0.5: epsilon *= 0.95
            # elif acc_rate_burn > 0.8: epsilon *= 1.05
            # print(f"Burn-in iter {i+1}/{burn_in}, Acc Rate: {acc_rate_burn:.3f}, New Eps: {epsilon:.6f}, LogProb: {current_log_prob.item():.2f}")
            print(f"Burn-in iter {i+1}/{burn_in}, Acc Rate (approx): {acc_rate_burn:.3f}, LogProb: {current_log_prob.item():.2f}")


print("\nRunning sampling...")
accepted_count_sample = 0
for i in tqdm(range(num_samples)):
    current_q, current_log_prob, accept_prob = hmc_step(current_q, log_prob_grad_fn, epsilon, L)
    samples_q[i, :] = current_q
    log_probs[i] = current_log_prob
    acceptance_rates.append(accept_prob)
    accepted_count_sample += (accept_prob > 0)

end_time_np = time.time()
total_time_np = end_time_np - start_time_np
avg_accept_rate = accepted_count_sample / num_samples

print(f"\nHMC sampling finished in {total_time_np:.1f} seconds.")
print(f"Average acceptance rate (sampling phase, approx): {avg_accept_rate:.3f}")
if avg_accept_rate < 0.1 or avg_accept_rate > 0.9:
     print("WARNING: Acceptance rate is very low or high. Epsilon/L tuning is likely needed.")

# %% 6. Post-Sampling Analysis & Uncertainty Quantification

print("\n--- Posterior Analysis & Uncertainty Quantification ---")

# Extract samples (convert back to constrained space where needed)
samples_params_unc = samples_q[:, :num_params]
samples_h = samples_q[:, num_params:]

samples_mu_h = samples_params_unc[:, 0].numpy()
samples_phi = transform_phi(samples_params_unc[:, 1]).numpy()
samples_sigma_h = transform_sigma_h(samples_params_unc[:, 2]).numpy()
samples_nu = transform_nu(samples_params_unc[:, 3]).numpy()

# --- Parameter Uncertainty ---
print("\nParameter Posterior Summaries:")
param_summary = pd.DataFrame({
    'Mean': [np.mean(samples_mu_h), np.mean(samples_phi), np.mean(samples_sigma_h), np.mean(samples_nu)],
    'Median': [np.median(samples_mu_h), np.median(samples_phi), np.median(samples_sigma_h), np.median(samples_nu)],
    'SD': [np.std(samples_mu_h), np.std(samples_phi), np.std(samples_sigma_h), np.std(samples_nu)],
    '5%': [np.percentile(samples_mu_h, 5), np.percentile(samples_phi, 5), np.percentile(samples_sigma_h, 5), np.percentile(samples_nu, 5)],
    '95%': [np.percentile(samples_mu_h, 95), np.percentile(samples_phi, 95), np.percentile(samples_sigma_h, 95), np.percentile(samples_nu, 95)],
}, index=['mu_h', 'phi', 'sigma_h', 'nu'])
print(param_summary)

# Plot parameter posteriors (Histograms/KDEs)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(samples_mu_h, kde=True, ax=axes[0, 0]).set_title('Posterior mu_h')
sns.histplot(samples_phi, kde=True, ax=axes[0, 1]).set_title('Posterior phi')
sns.histplot(samples_sigma_h, kde=True, ax=axes[1, 0]).set_title('Posterior sigma_h')
sns.histplot(samples_nu, kde=True, ax=axes[1, 1]).set_title('Posterior nu')
plt.tight_layout()
plt.show()
plt.close()

# Plot parameter trace plots (visual convergence check)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(samples_mu_h, alpha=0.7); axes[0].set_ylabel('mu_h')
axes[1].plot(samples_phi, alpha=0.7); axes[1].set_ylabel('phi')
axes[2].plot(samples_sigma_h, alpha=0.7); axes[2].set_ylabel('sigma_h')
axes[3].plot(samples_nu, alpha=0.7); axes[3].set_ylabel('nu')
axes[-1].set_xlabel('Iteration')
fig.suptitle('Parameter Trace Plots')
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle
plt.show()
plt.close()


# --- Latent Volatility Path Uncertainty ---
print("\nLatent Volatility Path Analysis:")
samples_ann_vol = np.exp(samples_h.numpy() / 2.0) * np.sqrt(252)

# Calculate mean and credible intervals (e.g., 90% CI)
mean_path = np.mean(samples_ann_vol, axis=0)
lower_ci = np.percentile(samples_ann_vol, 5, axis=0)
upper_ci = np.percentile(samples_ann_vol, 95, axis=0)

# Plot mean path with CI band AND individual posterior draws
plt.figure(figsize=(14, 7))
num_paths_to_plot = 50
indices = np.random.choice(num_samples, num_paths_to_plot, replace=False)
plt.plot(time_index, samples_ann_vol[indices, :].T, color='lightblue', alpha=0.1) # Individual paths
plt.plot(time_index, mean_path, color='darkblue', label='Posterior Mean Ann. Vol.')
# plt.fill_between(time_index, lower_ci, upper_ci, color='skyblue', alpha=0.4, label='90% Credible Interval') # Optional fill

plt.title(f'{ticker} Estimated Annualized Volatility (Showing Path Uncertainty)')
plt.ylabel('Annualized Volatility')
plt.xlabel('Date')
plt.legend(['Posterior Draws (Sample)', 'Posterior Mean']) # Simplified legend
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

print(f"Visualizing {num_paths_to_plot} posterior draws of the volatility path shows the functional uncertainty.")


# --- Bayesian VaR Uncertainty ---
print("\nValue-at-Risk (1-day ahead) Uncertainty:")
alpha_levels = [0.05, 0.01]

# Get last day's volatility scale samples
last_h_samples = samples_h[:, -1].numpy()
last_scale_samples = np.exp(last_h_samples / 2.0)
nu_samples = samples_nu # Already numpy array

plt.figure(figsize=(10, 4 * len(alpha_levels)))
for i, alpha in enumerate(alpha_levels):
    # Calculate VaR for each posterior sample using t-distribution ppf
    var_samples = stats.t.ppf(alpha, df=nu_samples, loc=0, scale=last_scale_samples)

    # Analyze and plot the posterior distribution of VaR(alpha)
    mean_var = np.mean(var_samples)
    hdi_var = np.percentile(var_samples, [3, 97]) # 94% HDI approx

    print(f"\nPosterior Distribution for {1-alpha:.0%} VaR:")
    print(f"  Mean VaR: {mean_var:.4f}")
    print(f"  94% HDI: [{hdi_var[0]:.4f}, {hdi_var[1]:.4f}]")

    plt.subplot(len(alpha_levels), 1, i + 1)
    sns.histplot(var_samples, kde=True, bins=50, stat='density')
    plt.axvline(mean_var, color='r', linestyle='--', label=f'Mean: {mean_var:.4f}')
    plt.axvline(hdi_var[0], color='g', linestyle=':', label='94% HDI')
    plt.axvline(hdi_var[1], color='g', linestyle=':')
    plt.title(f'Posterior Distribution of 1-Day {1-alpha:.0%} VaR')
    plt.xlabel('VaR Estimate (Log Return)')
    plt.legend()

plt.tight_layout()
plt.show()
plt.close()
print("The histograms above show the full uncertainty distribution for the VaR estimates.")


# --- Forecast Uncertainty (Volatility & Returns, e.g., 1 step ahead) ---
print("\nForecast Uncertainty (1-step ahead):")

# Simulate h_{T+1} for each posterior sample
mu_h_samples = samples_mu_h
phi_samples = samples_phi
sigma_h_samples = samples_sigma_h
h_T_samples = samples_h[:, -1].numpy() # Last observed h

# h_{T+1} = mu_h + phi * (h_T - mu_h) + sigma_h * eta_{T+1}
eta_T1 = np.random.randn(num_samples)
h_T1_samples = mu_h_samples + phi_samples * (h_T_samples - mu_h_samples) + sigma_h_samples * eta_T1
ann_vol_T1_samples = np.exp(h_T1_samples / 2.0) * np.sqrt(252)

# Simulate r_{T+1} ~ StudentT(nu, 0, exp(h_{T+1}/2))
scale_T1_samples = np.exp(h_T1_samples / 2.0)
# Need standard T random variables, then scale and shift
std_t_rvs = stats.t.rvs(df=nu_samples, size=num_samples)
r_T1_samples = 0 + scale_T1_samples * std_t_rvs # Since mu=0

# Plot forecast distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(ann_vol_T1_samples, kde=True, ax=axes[0], bins=50)
axes[0].set_title('Posterior Predictive Distribution: Ann. Vol (T+1)')
axes[0].set_xlabel('Annualized Volatility')

sns.histplot(r_T1_samples, kde=True, ax=axes[1], bins=50)
axes[1].set_title('Posterior Predictive Distribution: Return (T+1)')
axes[1].set_xlabel('Log Return')

plt.tight_layout()
plt.show()
plt.close()

print("Generated forecast distributions for next-day volatility and return, incorporating all parameter and latent state uncertainty.")
print(f"Forecast Mean Ann Vol (T+1): {np.mean(ann_vol_T1_samples):.4f}")
print(f"Forecast Mean Return (T+1): {np.mean(r_T1_samples):.6f}")
print(f"Forecast 90% CI Ann Vol (T+1): [{np.percentile(ann_vol_T1_samples, 5):.4f}, {np.percentile(ann_vol_T1_samples, 95):.4f}]")
print(f"Forecast 90% CI Return (T+1): [{np.percentile(r_T1_samples, 5):.4f}, {np.percentile(r_T1_samples, 95):.4f}]")

# %% 7. Conclusions on Uncertainty

print("\n--- Uncertainty Quantification Summary ---")
print("The Bayesian approach inherently quantifies uncertainty:")
print("1.  **Parameter Uncertainty:** Posterior distributions (plots above) for `mu_h`, `phi`, `sigma_h`, `nu` show the range of plausible values consistent with the data and priors. The width of these distributions reflects our uncertainty about the true parameter values.")
print(f"    - E.g., the 90% credible interval for persistence `phi` is [{param_summary.loc['phi', '5%']:.3f}, {param_summary.loc['phi', '95%']:.3f}].")
print(f"    - The degrees of freedom `nu` has a 90% CI of [{param_summary.loc['nu', '5%']:.2f}, {param_summary.loc['nu', '95%']:.2f}], indicating uncertainty about the exact 'fatness' of the tails.")
print("2.  **Latent State Uncertainty:** The plot showing multiple sampled volatility paths explicitly visualizes the uncertainty in the *historical* volatility trajectory. We don't know the exact path, only a distribution over possible paths.")
print("3.  **Measurement/Observation Uncertainty:** The Student's t-likelihood acknowledges that returns deviate randomly from the level set by the current volatility (captured by `nu`).")
print("4.  **Risk Measure Uncertainty:** The posterior distribution for VaR (histograms above) directly quantifies the uncertainty in our risk estimate. Instead of a single VaR number, we have a range of likely VaR values.")
print("5.  **Forecast Uncertainty:** Predictive distributions for future volatility and returns (plots above) combine uncertainty from all parameters and the stochastic nature of the model, providing a probabilistic range for future outcomes, not just a point forecast.")
print("6.  **Limitations:** This implementation uses basic HMC and requires careful tuning. Model uncertainty (e.g., comparing SV-t vs GARCH vs SV-Normal) is not addressed here but is another layer of uncertainty in real-world analysis.")