import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Still needed for distributions

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load Data (Old Faithful) - Same as before
try:
    data = pd.read_csv('https://raw.githubusercontent.com/stantinor/datasets/master/old_faithful.csv')
    data['eruptions_std'] = (data['eruptions'] - data['eruptions'].mean()) / data['eruptions'].std()
    data['waiting_std'] = (data['waiting'] - data['waiting'].mean()) / data['waiting'].std()
    eruptions = data['eruptions_std'].values
    waiting = data['waiting_std'].values
    print("Data loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}")
    print("Using dummy data for demonstration.")
    eruptions = np.random.rand(100) * 3 + 1.5
    waiting = 50 + 10 * eruptions + np.random.randn(100) * 5
    eruptions = (eruptions - eruptions.mean()) / eruptions.std()
    waiting = (waiting - waiting.mean()) / waiting.std()

n_samples_data = len(eruptions) # Store number of data points

# 2. Define the Log-Posterior Function (using NumPy)

def log_likelihood_np(params, x, y):
    """Calculates log-likelihood of data given parameters using NumPy."""
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)
    if sigma <= 0: # Safety check, though log_sigma makes sigma always positive
        return -np.inf
    mu = alpha + beta * x
    # Sum of log probabilities of Normal distribution
    ll = np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    # Check for NaN/Inf which can happen if sigma is extremely small/large
    if np.isnan(ll) or np.isinf(ll):
        return -np.inf
    return ll

def log_prior_np(params):
    """Calculates log-prior probability of parameters using NumPy."""
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma) # Transform back to sigma scale

    # Prior for alpha: Normal(0, 20)
    log_prior_alpha = stats.norm.logpdf(alpha, loc=0, scale=20)
    # Prior for beta: Normal(0, 10)
    log_prior_beta = stats.norm.logpdf(beta, loc=0, scale=10)
    # Prior for sigma: HalfCauchy(0, 5)
    if sigma <= 0:
        return -np.inf # Log probability is -infinity if sigma is not positive
    # logpdf(sigma | loc=0, scale=5) = log(2 / (pi * scale * (1 + (x/scale)^2))) for x > 0
    scale_cauchy = 5.0
    log_prior_sigma = np.log(2.) - np.log(np.pi * scale_cauchy * (1. + (sigma / scale_cauchy)**2))

    return log_prior_alpha + log_prior_beta + log_prior_sigma

def log_posterior_np(params, x, y):
    """Calculates log-posterior (unnormalized) using NumPy."""
    lp = log_prior_np(params)
    # Avoid calculating likelihood if prior is impossible
    if lp == -np.inf:
        return -np.inf
    ll = log_likelihood_np(params, x, y)
    if ll == -np.inf:
        return -np.inf
    return lp + ll

# 3. Define Manual Gradient Function for Log Posterior

def manual_grad_log_posterior(params, x, y):
    """
    Computes the gradient of the log posterior manually using NumPy.
    Gradient is with respect to [alpha, beta, log_sigma].
    """
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)
    n = len(x) # Number of data points

    # --- Gradient of Log Prior ---
    # d(log_prior_alpha)/d(alpha) = -(alpha - 0) / 20^2
    grad_log_prior_alpha = -alpha / (20**2)
    # d(log_prior_beta)/d(beta) = -(beta - 0) / 10^2
    grad_log_prior_beta = -beta / (10**2)
    # d(log_prior_sigma)/d(log_sigma) = d(log_prior_sigma)/d(sigma) * d(sigma)/d(log_sigma)
    # d(sigma)/d(log_sigma) = sigma
    # d(log_prior_sigma)/d(sigma) = -2*sigma / (scale^2 + sigma^2) [scale=5]
    scale_cauchy = 5.0
    if sigma <= 0:
         # Gradient is technically undefined, but step should be rejected by energy
         # Return 0 or a large value pushing away? Let's try 0, rely on energy check.
         grad_log_prior_log_sigma = 0.0
    else:
        grad_log_prior_log_sigma = (-2.0 * sigma**2 / (scale_cauchy**2 + sigma**2))

    grad_log_prior = np.array([grad_log_prior_alpha, grad_log_prior_beta, grad_log_prior_log_sigma])

    # --- Gradient of Log Likelihood ---
    if sigma <= 0:
        # If sigma is invalid, likelihood is -inf, gradient contribution should be zero
        # as the energy check will reject the step anyway.
        return grad_log_prior # Or np.array([0., 0., 0.])? Let's return prior gradient.

    mu = alpha + beta * x
    residuals = y - mu
    sigma_sq = sigma**2

    # d(LL)/d(alpha) = sum( (y_i - mu_i) / sigma^2 )
    grad_ll_alpha = np.sum(residuals / sigma_sq)
    # d(LL)/d(beta) = sum( x_i * (y_i - mu_i) / sigma^2 )
    grad_ll_beta = np.sum(x * residuals / sigma_sq)
    # d(LL)/d(log_sigma) = sum( (y_i - mu_i)^2 / sigma^2 - 1 )
    grad_ll_log_sigma = np.sum(residuals**2 / sigma_sq - 1.0)

    grad_log_likelihood = np.array([grad_ll_alpha, grad_ll_beta, grad_ll_log_sigma])

    # --- Total Gradient ---
    # Check for NaN/Inf in gradients (can happen with extreme sigma values)
    if np.any(np.isnan(grad_log_likelihood)) or np.any(np.isinf(grad_log_likelihood)):
         # Treat as zero gradient; step rejection should handle invalid parameters
         grad_log_likelihood = np.zeros_like(grad_log_likelihood)

    return grad_log_prior + grad_log_likelihood


# --- Visualization Helper (Unchanged) ---
def plot_results(samples, param_names=['alpha', 'beta', 'sigma']):
    """Plots traceplots and histograms of MCMC samples."""
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    plot_samples = samples.copy()
    # Transform log_sigma back to sigma for plotting if present
    try:
        sigma_idx = param_names.index('log_sigma')
        param_names_plot = param_names.copy() # Avoid modifying original list
        param_names_plot[sigma_idx] = 'sigma' # Rename for plot
        plot_samples[:, sigma_idx] = np.exp(plot_samples[:, sigma_idx])
    except ValueError: # If 'log_sigma' not in names (e.g., Gibbs output sigma)
        param_names_plot = param_names

    for i in range(n_params):
        ax_trace = axes[i, 0]
        ax_hist = axes[i, 1]

        # Trace plot
        ax_trace.plot(plot_samples[:, i], alpha=0.7)
        ax_trace.set_ylabel(param_names_plot[i])
        ax_trace.set_title(f'Trace Plot - {param_names_plot[i]}')

        # Histogram
        sns.histplot(plot_samples[:, i], kde=True, ax=ax_hist)
        ax_hist.set_title(f'Posterior Histogram - {param_names_plot[i]}')

    axes[-1, 0].set_xlabel('Iteration')
    axes[-1, 1].set_xlabel('Parameter Value')
    fig.tight_layout()
    plt.show()


# --- Metropolis-Hastings Implementation (NumPy compatible) ---
def metropolis_hastings(log_posterior_fn, initial_params, n_iter, proposal_sd, data_x, data_y):
    """Performs Metropolis-Hastings sampling using NumPy."""
    print("Running Metropolis-Hastings...")
    current_params = np.array(initial_params)
    n_params = len(current_params)
    samples = np.zeros((n_iter, n_params))
    log_post_current = log_posterior_fn(current_params, data_x, data_y)

    # Handle case where initial point is invalid
    if log_post_current == -np.inf:
        print("Warning: Initial parameters have log probability -inf. Trying random walk.")
        # Try a few random steps to find a valid starting point (simple fix)
        for _ in range(100):
             current_params = np.random.normal(loc=initial_params, scale=proposal_sd*5)
             log_post_current = log_posterior_fn(current_params, data_x, data_y)
             if log_post_current != -np.inf:
                 print("Found a valid starting point.")
                 break
        if log_post_current == -np.inf:
             print("Error: Could not find valid starting parameters. Check priors/initial guess.")
             return None, 0.0


    accepted_count = 0
    for i in range(n_iter):
        proposal = np.random.normal(loc=current_params, scale=proposal_sd, size=n_params)
        log_post_proposal = log_posterior_fn(proposal, data_x, data_y)

        log_acceptance_ratio = log_post_proposal - log_post_current

        # Acceptance probability calculation using NumPy checks
        if log_post_proposal == -np.inf:
            acceptance_prob = 0.0 # Never accept invalid proposal
        elif log_post_current == -np.inf:
             acceptance_prob = 1.0 # Always accept if moving from invalid state
        else:
            acceptance_prob = np.exp(min(0, log_acceptance_ratio)) # Equivalent to min(1, exp(ratio))

        u = np.random.rand()
        if u < acceptance_prob:
            current_params = proposal
            log_post_current = log_post_proposal
            accepted_count += 1

        samples[i] = current_params
        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    acceptance_rate = accepted_count / n_iter
    print(f"Metropolis-Hastings finished. Acceptance rate: {acceptance_rate:.3f}")
    return samples, acceptance_rate

# --- Gibbs Sampling Implementation (Unchanged - Already NumPy/SciPy) ---
# (Requires conjugate priors as defined before)
def gibbs_sampler(initial_params_gibbs, n_iter, data_x, data_y, prior_params):
    """Performs Gibbs sampling (NumPy/SciPy based)."""
    print("\nRunning Gibbs Sampling (with conjugate priors)...")
    alpha, beta, sigma_sq = initial_params_gibbs
    n_samples = len(data_x)
    samples = np.zeros((n_iter, 3)) # Store alpha, beta, sigma_sq

    x_sum = np.sum(data_x); y_sum = np.sum(data_y)
    x_sq_sum = np.sum(data_x**2); xy_sum = np.sum(data_x * data_y)
    alpha_mean_prior, alpha_var_prior = prior_params['alpha_mean'], prior_params['alpha_var']
    beta_mean_prior, beta_var_prior = prior_params['beta_mean'], prior_params['beta_var']
    a_prior, b_prior = prior_params['sigma_sq_a'], prior_params['sigma_sq_b']
    alpha_prec_prior = 1.0 / alpha_var_prior
    beta_prec_prior = 1.0 / beta_var_prior

    for i in range(n_iter):
        # Sample alpha
        alpha_var_cond = 1.0 / (alpha_prec_prior + n_samples / sigma_sq)
        alpha_mean_cond = alpha_var_cond * (alpha_prec_prior * alpha_mean_prior + (y_sum - beta * x_sum) / sigma_sq)
        alpha = np.random.normal(loc=alpha_mean_cond, scale=np.sqrt(alpha_var_cond))
        # Sample beta
        beta_var_cond = 1.0 / (beta_prec_prior + x_sq_sum / sigma_sq)
        beta_mean_cond = beta_var_cond * (beta_prec_prior * beta_mean_prior + (xy_sum - alpha * x_sum) / sigma_sq)
        beta = np.random.normal(loc=beta_mean_cond, scale=np.sqrt(beta_var_cond))
        # Sample sigma_sq
        residuals = data_y - (alpha + beta * data_x)
        ssr = np.sum(residuals**2)
        a_cond = a_prior + n_samples / 2.0
        b_cond = b_prior + ssr / 2.0
        sigma_sq = 1.0 / np.random.gamma(shape=a_cond, scale=1.0 / b_cond)

        samples[i] = [alpha, beta, sigma_sq]
        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    print("Gibbs Sampling finished.")
    samples_final = samples.copy()
    samples_final[:, 2] = np.sqrt(samples_final[:, 2]) # Convert sigma_sq to sigma
    return samples_final


# --- Slice Sampling Implementation (NumPy compatible) ---
def slice_sampler_1d(log_target_pdf, x_current, w, args=()):
    """Performs one step of univariate slice sampling using NumPy."""
    log_fx_current = log_target_pdf(x_current, *args)
    # Check if current point is valid
    if not np.isfinite(log_fx_current):
        # This shouldn't happen if the previous state was valid, but as a fallback...
        print(f"Warning: log_target_pdf returned {log_fx_current} for x_current={x_current}. Returning x_current.")
        return x_current

    log_u = log_fx_current - np.random.exponential(1.0)

    r = np.random.uniform(0, w)
    L = x_current - r
    R = x_current + (w - r)

    # Stepping out
    # Add max iterations to prevent infinite loops if target is weird
    max_steps = 100
    steps_out = 0
    while log_target_pdf(L, *args) > log_u and steps_out < max_steps:
        L -= w
        steps_out +=1
    steps_out = 0
    while log_target_pdf(R, *args) > log_u and steps_out < max_steps:
        R += w
        steps_out +=1

    # Shrinking in
    max_shrinks = 100
    shrinks = 0
    while shrinks < max_shrinks:
        x_new = np.random.uniform(L, R)
        log_fx_new = log_target_pdf(x_new, *args)

        if log_fx_new > log_u:
            return x_new
        else:
            if x_new < x_current:
                L = x_new
            else:
                R = x_new
        shrinks += 1
    # If max shrinks reached, return current value (safer than getting stuck)
    # print("Warning: Slice sampler max shrinks reached. Returning current value.")
    return x_current


def slice_sampler(log_posterior_fn, initial_params, n_iter, w, data_x, data_y):
    """Performs Slice Sampling using NumPy."""
    print("\nRunning Slice Sampling...")
    current_params = np.array(initial_params)
    n_params = len(current_params)
    samples = np.zeros((n_iter, n_params))

    if isinstance(w, (int, float)):
        w = np.full(n_params, w)

    # Check initial point validity
    if log_posterior_fn(current_params, data_x, data_y) == -np.inf:
         print("Error: Initial parameters have log probability -inf for Slice Sampler.")
         return None

    for i in range(n_iter):
        for j in range(n_params):
            # Define a function for the log posterior varying only the j-th param
            # Use a lambda function for cleaner argument passing
            log_posterior_1d = lambda param_j, params_frozen, idx, x, y: \
                log_posterior_fn(np.concatenate((params_frozen[:idx], [param_j], params_frozen[idx+1:])), x, y)

            current_params[j] = slice_sampler_1d(
                log_posterior_1d,
                current_params[j],
                w[j],
                args=(current_params, j, data_x, data_y) # Pass necessary fixed args
            )

        samples[i] = current_params.copy() # Store a copy

        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    print("Slice Sampling finished.")
    return samples


# --- Hamiltonian Monte Carlo (HMC) Implementation (NumPy + Manual Gradient) ---

def leapfrog_np(q, p, grad_log_post_fn, epsilon, L, data_x, data_y):
    """Performs L leapfrog steps for HMC using NumPy."""
    q_new = q.copy()
    p_new = p.copy()
    grad = grad_log_post_fn(q_new, data_x, data_y)

    # Half step for momentum at the beginning
    p_new += epsilon / 2.0 * grad

    # Alternate full steps for position and momentum
    for _ in range(L - 1):
        # Full step for position (q)
        q_new += epsilon * p_new # Assuming mass matrix M=I (p = M*dq/dt => dq/dt = M^-1*p = p)
        # Full step for momentum (p)
        grad = grad_log_post_fn(q_new, data_x, data_y)
        # Check if gradient calculation failed (e.g., invalid params in log_post)
        if np.any(np.isnan(grad)):
             print("Warning: NaN gradient in leapfrog step. Stopping trajectory.")
             return q, p # Return original position and momentum to likely reject step
        p_new += epsilon * grad

    # Final full step for position
    q_new += epsilon * p_new
    # Final half step for momentum
    grad = grad_log_post_fn(q_new, data_x, data_y)
    if np.any(np.isnan(grad)):
         print("Warning: NaN gradient in leapfrog step (final). Stopping trajectory.")
         return q, p
    p_new += epsilon / 2.0 * grad

    # Negate momentum at end of trajectory to make proposal symmetric
    p_new = -p_new

    return q_new, p_new

def hamiltonian_monte_carlo_np(log_posterior_fn, grad_log_post_fn, initial_params,
                               n_iter, epsilon, L, data_x, data_y):
    """Performs Hamiltonian Monte Carlo sampling using NumPy and manual gradient."""
    print("\nRunning Hamiltonian Monte Carlo (HMC) with NumPy...")
    current_q = np.array(initial_params)
    n_params = len(current_q)
    samples = np.zeros((n_iter, n_params))

    # Evaluate log posterior and potential energy at the start
    current_log_post = log_posterior_fn(current_q, data_x, data_y)
    current_U = -current_log_post

    # Check initial point validity
    if not np.isfinite(current_U):
        print("Error: Initial parameters have non-finite potential energy (log_posterior is inf or -inf). Cannot start HMC.")
        return None, 0.0

    accepted_count = 0
    for i in range(n_iter):
        # 1. Sample momentum (p)
        current_p = np.random.normal(size=n_params)

        # 2. Simulate Hamiltonian dynamics using leapfrog
        q_proposal, p_proposal = leapfrog_np(current_q, current_p, grad_log_post_fn,
                                             epsilon, L, data_x, data_y)

        # 3. Calculate Metropolis-Hastings acceptance probability
        proposal_log_post = log_posterior_fn(q_proposal, data_x, data_y)
        proposal_U = -proposal_log_post

        # Check for valid states (finite potential energy)
        if not np.isfinite(proposal_U):
            # Proposed state is invalid (e.g., sigma <= 0), always reject
            acceptance_prob = 0.0
        else:
            # Calculate kinetic energy (assuming mass matrix M = I)
            current_K = 0.5 * np.sum(current_p**2)
            proposal_K = 0.5 * np.sum(p_proposal**2)

            # Calculate total energy (Hamiltonian)
            current_H = current_U + current_K
            proposal_H = proposal_U + proposal_K

            # Acceptance probability (log scale for stability)
            log_acceptance_ratio = current_H - proposal_H # Equivalent to log(pi(q')pi(p') / pi(q)pi(p))

            acceptance_prob = np.exp(min(0.0, log_acceptance_ratio))

        # 4. Accept or reject
        u = np.random.rand()
        if u < acceptance_prob:
            current_q = q_proposal
            current_U = proposal_U # Update potential energy
            current_log_post = proposal_log_post # Update log posterior
            accepted_count += 1
        # Else: current_q and current_U remain the same

        samples[i] = current_q

        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    acceptance_rate = accepted_count / n_iter
    print(f"HMC finished. Acceptance rate: {acceptance_rate:.3f}")
    return samples, acceptance_rate


# --- Run MH ---
print("--- Running Metropolis-Hastings ---")
n_iterations_mh = 20000
burn_in_mh = 5000
initial_guess_mh = [0.0, 0.0, np.log(np.std(waiting))] # Use log_sigma internally
proposal_std_mh = [0.05, 0.05, 0.05] # Tune this!

mh_samples, mh_acceptance_rate = metropolis_hastings(
    log_posterior_np, initial_guess_mh, n_iterations_mh, proposal_std_mh, eruptions, waiting
)

if mh_samples is not None:
    mh_samples_burned = mh_samples[burn_in_mh:]
    plot_results(mh_samples_burned, param_names=['alpha', 'beta', 'log_sigma'])
    print(f"MH Acceptance Rate: {mh_acceptance_rate:.3f}")

# --- Run Gibbs (using conjugate priors for demonstration) ---
print("\n--- Running Gibbs Sampling ---")
n_iterations_gibbs = 20000
burn_in_gibbs = 5000
conjugate_priors = {
    'alpha_mean': 0.0, 'alpha_var': 100.0,
    'beta_mean': 0.0, 'beta_var': 100.0,
    'sigma_sq_a': 0.01, 'sigma_sq_b': 0.01
}
initial_guess_gibbs = [0.0, 0.0, np.var(waiting)] # Start variance at sample variance

gibbs_samples = gibbs_sampler(
    initial_guess_gibbs, n_iterations_gibbs, eruptions, waiting, conjugate_priors
)
if gibbs_samples is not None:
    gibbs_samples_burned = gibbs_samples[burn_in_gibbs:]
    # Gibbs outputs sigma directly, not log_sigma
    plot_results(gibbs_samples_burned, param_names=['alpha', 'beta', 'sigma'])

# --- Run Slice Sampler ---
print("\n--- Running Slice Sampling ---")
n_iterations_slice = 20000
burn_in_slice = 5000
initial_guess_slice = [0.0, 0.0, np.log(np.std(waiting))]
slice_widths = [0.5, 0.5, 0.5] # Tune this!

slice_samples = slice_sampler(
    log_posterior_np, initial_guess_slice, n_iterations_slice, slice_widths, eruptions, waiting
)
if slice_samples is not None:
    slice_samples_burned = slice_samples[burn_in_slice:]
    plot_results(slice_samples_burned, param_names=['alpha', 'beta', 'log_sigma'])


# --- Run HMC (NumPy version) ---
print("\n--- Running Hamiltonian Monte Carlo (NumPy) ---")
n_iterations_hmc = 10000 # Fewer iterations often needed
burn_in_hmc = 2000
initial_guess_hmc = [0.0, 0.0, np.log(np.std(waiting))]
hmc_epsilon = 0.05  # Tune this! Smaller might be needed for manual gradients
hmc_L = 20          # Tune this!

hmc_samples_np, hmc_acceptance_rate_np = hamiltonian_monte_carlo_np(
    log_posterior_np, manual_grad_log_posterior, initial_guess_hmc, n_iterations_hmc,
    hmc_epsilon, hmc_L, eruptions, waiting
)

if hmc_samples_np is not None:
    hmc_samples_np_burned = hmc_samples_np[burn_in_hmc:]
    plot_results(hmc_samples_np_burned, param_names=['alpha', 'beta', 'log_sigma'])
    print(f"HMC (NumPy) Acceptance Rate: {hmc_acceptance_rate_np:.3f}")

print("\n--- NUTS Explanation (Still Relevant) ---")
print("The No-U-Turn Sampler (NUTS) adaptively chooses the number of leapfrog steps (L)")
print("based on trajectory simulation, avoiding manual tuning of L and improving efficiency.")
print("Implementing NUTS requires complex tree-building logic and is best handled by libraries.")