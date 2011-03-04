"""Simple Monte Carlo integration to compute the data likelihoods and
the Bayes factor of the multi-subject Beta-Binomial model.

Copyright Emanuele Olivetti, 2011.

This program is distributed under the GNU General Public Licence v3.0.

Some useful formulas on parameters of the Beta distribution:
Beta(a,b):
  mu = a/(a+b)
  sigma2 = a*b/((a+b)**2 * (a+b+1))
or:
  a = mu * (mu - mu**2 - sigma2) / sigma2
  b = mu * (mu**2 - 2*mu + sigma2 + 1) / sigma2 - 1
and
  a > 0  <->  0 < sigma2 < mu*(1-mu)
When a=b:
  mu = 0.5
  sigma2 = 1 / (4*(2*a+1))
or
  a = 1/(8*sigma2) - 0.5
and
  a > 0  <->  0 < sigma2 < 0.25
"""

import numpy as np
from scipy.special import beta, gamma, betaln, gammaln
from scipy import comb

def beta_binomial(k, n, a, b):
    """The pmf/pdf of the Beta-binomial distribution.

    Computation based on beta function.

    See: http://en.wikipedia.org/wiki/Beta-binomial_distribution

    k = a vector of non-negative integers <= n
    n = an integer
    a = an array of non-negative real numbers
    b = an array of non-negative real numbers
    """
    return (comb(n, k) * beta(k+a, n-k+b) / beta(a,b)).prod(0)


def beta_binomial_gammaln(k, n, a, b):
    """The log of the pmf/pdf of the Beta-binomial distribution.
    Computation based on gammaln function.

    Note: this implementation suffers much less the numerical issues
    of beta_binomial().

    See: http://en.wikipedia.org/wiki/Beta-binomial_distribution

    k = a vector of non-negative integers <= n
    n = an integer
    a = an array of non-negative real numbers
    b = an array of non-negative real numbers
    """
    tmp0 = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))
    tmp1 = gammaln(a+k) + gammaln(n+b-k)
    tmp2 = - gammaln(a+b+n) + gammaln(a+b) - (gammaln(a) + gammaln(b))
    return np.exp((tmp0 + tmp1 + tmp2).sum(0))


def a_from_sigma(sigma):
    """Compute the parameter a of a Beta distribution given mu=1/2 and
    sigma.
    """
    a = 1.0/(8.0*(sigma*sigma)) - 0.5
    return a


def H0_sigma_beta_binomial(m, iterations=1e5):
    """H0: population is parametrized via sigma, as suggested in
    Gelman 2006.
    """
    mu = 0.5
    sigma = np.random.uniform(low=0, high=0.5, size=iterations)
    a = a_from_sigma(sigma)
    return a, sigma


def a_b_from_mu_sigma(mu, sigma):
    """Compute the parameters (a, b) of a Beta distribution given
    mu and sigma.
    """
    mu2 = mu*mu
    sigma2 = sigma*sigma
    a = mu*(-mu2 + mu - sigma2)/sigma2
    b = mu * (mu2 - 2*mu + sigma2 + 1) / sigma2 - 1
    return a, b


def H1_sigma_beta_binomial(m, iterations=1e5):
    """H1: population is parametrized via mu and sigma.
    """
    mu = np.random.uniform(low=0, high=0.5, size=iterations)
    sigma_upper = np.sqrt(mu*(1.0-mu))
    sigma = np.random.uniform(low=0.0, high=sigma_upper, size=iterations)
    a, b = a_b_from_mu_sigma(mu, sigma)
    return a, b, mu, sigma


def bayes_factor(errors, m, iterations):
    """Compute Bayes factor.
    """
    N = len(errors)

    a, sigma0 = H0_sigma_beta_binomial(m=m, iterations=iterations)
    p_errors_given_H0 = beta_binomial_gammaln(errors[:,None], m, a, a)
    
    a, b, mu1, sigma1 = H1_sigma_beta_binomial(m=m, iterations=iterations)
    p_errors_given_H1 = beta_binomial_gammaln(errors[:,None], m, a, b)

    B_10 = p_errors_given_H1.mean() / p_errors_given_H0.mean()
    return B_10, p_errors_given_H1, p_errors_given_H0, sigma0, mu1, sigma1


if __name__=='__main__':

    iterations = 1e5

    N = 15
    m = 108
    errors = np.array([43, 59, 51, 38, 39, 53, 47, 50, 50, 59, 59, 45, 36, 46, 53])

    B_10, p_errors_given_H1, p_errors_given_H0, a0, a1, b1 = bayes_factor(errors, m, iterations)
    print "Observed errors:", errors
    print "Beta-binomial hierarchical model:"
    print "\t Simple Monte Carlo, iterations =", iterations
    print "\t p(errors|H0) =", p_errors_given_H0.mean()
    print "\t p(errors|H1) =", p_errors_given_H1.mean()
    print "\t B_10 =", B_10
    
