import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jax.numpy as jnp # For HMC gradients
from jax import grad

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load Data (Old Faithful)
# Using a readily available version online
try:
    data = pd.read_csv('https://raw.githubusercontent.com/stantinor/datasets/master/old_faithful.csv')
    # Use standardized versions for better numerical stability
    data['eruptions_std'] = (data['eruptions'] - data['eruptions'].mean()) / data['eruptions'].std()
    data['waiting_std'] = (data['waiting'] - data['waiting'].mean()) / data['waiting'].std()
    eruptions = data['eruptions_std'].values
    waiting = data['waiting_std'].values
    print("Data loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}")
    print("Please download 'old_faithful.csv' manually if needed.")
    # Example: provide link or instructions
    # As a fallback, create dummy data for demonstration
    print("Using dummy data for demonstration.")
    eruptions = np.random.rand(100) * 3 + 1.5 # Dummy eruption durations
    waiting = 50 + 10 * eruptions + np.random.randn(100) * 5 # Dummy waiting times
    # Standardize dummy data
    eruptions = (eruptions - eruptions.mean()) / eruptions.std()
    waiting = (waiting - waiting.mean()) / waiting.std()


# 2. Define the Log-Posterior Function
# We work with log(sigma) internally for easier sampling (unconstrained)
# but define priors and likelihood in terms of sigma.

def log_likelihood(params, x, y):
    """Calculates log-likelihood of data given parameters."""
    alpha, beta, log_sigma = params
    sigma = jnp.exp(log_sigma)
    mu = alpha + beta * x
    # Sum of log probabilities of Normal distribution
    ll = jnp.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return ll

def log_prior(params):
    """Calculates log-prior probability of parameters."""
    alpha, beta, log_sigma = params
    sigma = jnp.exp(log_sigma) # Transform back to sigma scale

    # Prior for alpha: Normal(0, 20) -> variance = 400
    log_prior_alpha = stats.norm.logpdf(alpha, loc=0, scale=20)
    # Prior for beta: Normal(0, 10) -> variance = 100
    log_prior_beta = stats.norm.logpdf(beta, loc=0, scale=10)
    # Prior for sigma: HalfCauchy(0, 5)
    # logpdf(sigma | loc=0, scale=5) = log(2 / (pi * scale * (1 + (x/scale)^2))) for x > 0
    if sigma <= 0:
        return -jnp.inf # Log probability is -infinity if sigma is not positive
    log_prior_sigma = jnp.log(2) - jnp.log(jnp.pi * 5 * (1 + (sigma / 5)**2))

    return log_prior_alpha + log_prior_beta + log_prior_sigma

# Use JAX numpy for compatibility with automatic differentiation if needed later
def log_posterior(params, x, y):
    """Calculates log-posterior (unnormalized)."""
    lp = log_prior(params)
    # Avoid calculating likelihood if prior is impossible
    if lp == -jnp.inf:
        return -jnp.inf
    ll = log_likelihood(params, x, y)
    return lp + ll

# --- Visualization Helper ---
def plot_results(samples, param_names=['alpha', 'beta', 'sigma']):
    """Plots traceplots and histograms of MCMC samples."""
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    # Transform log_sigma back to sigma for plotting
    plot_samples = samples.copy()
    if 'log_sigma' in param_names:
        sigma_idx = param_names.index('log_sigma')
        param_names[sigma_idx] = 'sigma' # Rename for plot
        plot_samples[:, sigma_idx] = np.exp(plot_samples[:, sigma_idx])

    for i in range(n_params):
        ax_trace = axes[i, 0]
        ax_hist = axes[i, 1]

        # Trace plot
        ax_trace.plot(plot_samples[:, i], alpha=0.7)
        ax_trace.set_ylabel(param_names[i])
        ax_trace.set_title(f'Trace Plot - {param_names[i]}')

        # Histogram
        sns.histplot(plot_samples[:, i], kde=True, ax=ax_hist)
        ax_hist.set_title(f'Posterior Histogram - {param_names[i]}')

    axes[-1, 0].set_xlabel('Iteration')
    axes[-1, 1].set_xlabel('Parameter Value')
    fig.tight_layout()
    plt.show()

# --- Gibbs Sampling Implementation (Using Conjugate Priors for Simplicity) ---
# NOTE: We redefine the priors and target function *only* for this Gibbs example.

def gibbs_sampler(initial_params_gibbs, n_iter, data_x, data_y, prior_params):
    """
    Performs Gibbs sampling for Bayesian linear regression with conjugate priors.

    Args:
        initial_params_gibbs: Starting values [alpha, beta, sigma_sq].
        n_iter: Number of MCMC iterations.
        data_x: Predictor variable data.
        data_y: Response variable data.
        prior_params: Dict containing prior hyperparameters {'alpha_mean', 'alpha_var',
                      'beta_mean', 'beta_var', 'sigma_sq_a', 'sigma_sq_b'}.

    Returns:
        numpy array of samples.
    """
    print("\nRunning Gibbs Sampling (with conjugate priors)...")
    alpha, beta, sigma_sq = initial_params_gibbs
    n_samples = len(data_x)
    samples = np.zeros((n_iter, 3)) # Store alpha, beta, sigma_sq

    # Precompute sums for efficiency
    x_sum = np.sum(data_x)
    y_sum = np.sum(data_y)
    x_sq_sum = np.sum(data_x**2)
    xy_sum = np.sum(data_x * data_y)

    # Prior hyperparameters
    alpha_mean_prior, alpha_var_prior = prior_params['alpha_mean'], prior_params['alpha_var']
    beta_mean_prior, beta_var_prior = prior_params['beta_mean'], prior_params['beta_var']
    a_prior, b_prior = prior_params['sigma_sq_a'], prior_params['sigma_sq_b']

    # Precompute prior precision terms
    alpha_prec_prior = 1.0 / alpha_var_prior
    beta_prec_prior = 1.0 / beta_var_prior

    for i in range(n_iter):
        # 1. Sample alpha | beta, sigma_sq, data
        # Conditional is Normal(mu_alpha, var_alpha)
        alpha_var_cond = 1.0 / (alpha_prec_prior + n_samples / sigma_sq)
        alpha_mean_cond = alpha_var_cond * (alpha_prec_prior * alpha_mean_prior +
                                            (y_sum - beta * x_sum) / sigma_sq)
        alpha = np.random.normal(loc=alpha_mean_cond, scale=np.sqrt(alpha_var_cond))

        # 2. Sample beta | alpha, sigma_sq, data
        # Conditional is Normal(mu_beta, var_beta)
        beta_var_cond = 1.0 / (beta_prec_prior + x_sq_sum / sigma_sq)
        beta_mean_cond = beta_var_cond * (beta_prec_prior * beta_mean_prior +
                                          (xy_sum - alpha * x_sum) / sigma_sq)
        beta = np.random.normal(loc=beta_mean_cond, scale=np.sqrt(beta_var_cond))

        # 3. Sample sigma_sq | alpha, beta, data
        # Conditional is InvGamma(a_cond, b_cond)
        residuals = data_y - (alpha + beta * data_x)
        ssr = np.sum(residuals**2)
        a_cond = a_prior + n_samples / 2.0
        b_cond = b_prior + ssr / 2.0
        # Sample from InvGamma(a, b) by sampling 1/Gamma(a, 1/b)
        sigma_sq = 1.0 / np.random.gamma(shape=a_cond, scale=1.0 / b_cond)

        # Store the updated parameters
        samples[i] = [alpha, beta, sigma_sq]

        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    print("Gibbs Sampling finished.")
    # Transform sigma_sq back to sigma for consistency in plotting
    samples_final = samples.copy()
    samples_final[:, 2] = np.sqrt(samples_final[:, 2]) # sigma = sqrt(sigma_sq)
    return samples_final

# --- Run Gibbs ---
n_iterations_gibbs = 20000
burn_in_gibbs = 5000

# Define conjugate priors (relatively uninformative)
# Normal(mean, variance) for alpha, beta
# InvGamma(shape, rate) for sigma^2
conjugate_priors = {
    'alpha_mean': 0.0, 'alpha_var': 100.0, # Corresponds to Normal(0, 10^2)
    'beta_mean': 0.0, 'beta_var': 100.0,   # Corresponds to Normal(0, 10^2)
    'sigma_sq_a': 0.01, 'sigma_sq_b': 0.01 # Weakly informative InvGamma prior
}

# Initial guess for Gibbs (alpha, beta, sigma_sq)
initial_guess_gibbs = [0.0, 0.0, np.var(waiting)] # Start variance at sample variance

gibbs_samples = gibbs_sampler(
    initial_guess_gibbs, n_iterations_gibbs, eruptions, waiting, conjugate_priors
)

# Discard burn-in and plot (parameter names match the output: alpha, beta, sigma)
gibbs_samples_burned = gibbs_samples[burn_in_gibbs:]
plot_results(gibbs_samples_burned, param_names=['alpha', 'beta', 'sigma'])
# Note: Gibbs has an implicit acceptance rate of 100% as it samples directly.