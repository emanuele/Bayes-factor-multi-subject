""" Compute posterior:
        p(mu, sigma | errors, H1)
        p(sigma | errors, H0)

Copyright Emanuele Olivetti, 2011.

This program is distributed under the GNU General Public Licence v3.0.

Procedure:
p(mu,sigma|errors) = p(errors|mu,sigma)*p(sigma|mu)*p(mu)/p(errors)
where:
p(sigma|mu) = 1/(sigma_upper - sigma_lower) = 1.0/sigma_upper
p(mu) = 1/(mu_lower - mu_upper) = 1/0.5 = 2.0
Normalization: p(errors) = \int p(errors|mu,sigma) p(mu,sigma) dmu dsigma
"""

import numpy as np
import bayes_factor


if __name__=='__main__':

    iterations = 1e4

    N = 15
    m = 108
    errors = np.array([43, 59, 51, 38, 39, 53, 47, 50, 50, 59, 59, 45, 36, 46, 53])
    # errors = np.linspace(0,60,N).astype(np.int)
    print "errors =", errors

    B_10, p_errors_given_mu_sigma, p_errors_given_sigma, sigma0, mu1, sigma1 = bayes_factor.bayes_factor(errors, m, iterations)

    sigma1_upper = np.sqrt(mu1 * (1.0 - mu1))
    p_mu_sigma_given_errors = p_errors_given_mu_sigma *  1.0/sigma1_upper * 2.0
    integral_mc1 =  p_errors_given_mu_sigma.mean()
    p_mu_sigma_given_errors /= integral_mc1

    p_sigma_given_errors = p_errors_given_sigma * 2.0
    integral_mc0 = p_errors_given_sigma.mean()
    p_sigma_given_errors /= integral_mc0

    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    size = 500
    sigma_i = np.linspace(0.0, 0.5, size)
    a = bayes_factor.a_from_sigma(sigma_i)
    p_sigma_given_errors_i = bayes_factor.beta_binomial_gammaln(errors[:,None], m, a, a) * 2.0 / integral_mc0
    plt.figure()
    plt.plot(sigma_i, p_sigma_given_errors_i, 'k-')
    plt.xlabel('$\sigma$')
    plt.ylabel('$p(\sigma)$')
    plt.title("$p(\sigma|e_1,\ldots,e_{%s},m=%s,H_0)$" % (N, m))
        
    size = 100
    mu_i = np.linspace(0, 0.5, size)
    sigma_i = np.linspace(0, 0.5, size)
    zi = mlab.griddata(mu1, sigma1, p_mu_sigma_given_errors, mu_i, sigma_i)
    plt.figure()
    C = plt.contourf(mu_i, sigma_i, zi, 20)
    plt.colorbar()
    plt.xlabel('$\mu$')
    plt.ylabel('$\sigma$')
    plt.title("$p(\mu,\sigma|e_1,\ldots,e_{%s},m=%s,H_1$)" % (N, m))
    
    plt.show()
