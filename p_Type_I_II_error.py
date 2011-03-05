"""Simulated experiments to compute p(Type I error) and p(Type II
error). The Beta-binomial model is assumed and simluated errors are
generated given the true mu and sigma of the Beta distribution of the
population. Given a range for the size m of the test sets or a range
for the number N of subjects the frequency of Type I (or Type II)
error are plotted.

Copyright Emanuele Olivetti, 2011.

This program is distributed under the GNU General Public Licence v3.0.

"""

import numpy as np
import bayes_factor

if __name__=='__main__':
    
    iterations = 1e4 # number of iterations to compute the Bayes factor.
    mu_true = 0.35 # true mu of the Beta distribution of the population.
    sigma_true = 0.15 # true sigma of the Beta distribution of the population.
    M = 50 # number of iterations to compute
    a, b = bayes_factor.a_b_from_mu_sigma(mu_true, sigma_true)
    # m_range = [10, 20, 50, 100]
    # N_range = [10]
    m_range = [50]
    N_range = [4, 8, 12, 16, 20, 30]

    print "m_range =", m_range
    print "N_range =", N_range
    print "mu =", mu_true
    print "sigma =", sigma_true
    print "M =", M

    BF = np.zeros((M, len(m_range), len(N_range)))
    for k, N in enumerate(N_range):
        for j, m in enumerate(m_range):
            for i in range(M):
                epsilon = np.random.beta(a=a, b=b, size=N)
                errors = np.random.binomial(n=m, p=epsilon)
                print i,') N =', N, ', m =', m
                print "\t errors:", errors
                B_10, p1, p0, a0, a1, b1 = bayes_factor.bayes_factor(errors, m, iterations)
                BF[i,j,k] = B_10
                print "\t p(errors|H0) =", p0.mean()    
                print "\t p(errors|H1) =", p1.mean()
                print "\t B_10 =", B_10
                print "\t B_01 =", 1.0/B_10

    BF_threshold = [1, 3, 20, 150]

    import matplotlib.pyplot as plt

    if mu_true==0.5:
        description = "p(Type I error)"
        ylabel = description
    else:
        description = "p(Type II error)"
        ylabel = description

    if len(m_range)>1:
        for k,N in enumerate(N_range):
            plt.figure()
            plt.title(description+" : N=%s, mu=%s, sigma=%s, M=%s" % (N, mu_true, sigma_true, M))
            for bft in BF_threshold:
                if mu_true==0.5:
                    tmp = np.mean(BF[:,:,k]>bft,0)
                else:
                    tmp = np.mean(BF[:,:,k]<bft,0)
                plt.plot(np.array(m_range),tmp,'o-', label='BF>%s' % bft)
            plt.legend()
            plt.ylabel(ylabel)
            plt.xlabel('m (testset size)')

    if len(N_range)>1:
        for j,m in enumerate(m_range):
            plt.figure()
            plt.title(description+" : m=%s, mu=%s, sigma=%s, M=%s" % (m, mu_true, sigma_true, M))
            for bft in BF_threshold:
                if mu_true==0.5:
                    tmp = np.mean(BF[:,j,:]>bft,0)
                else:
                    tmp = np.mean(BF[:,j,:]<bft,0)
                plt.plot(np.array(N_range),tmp,'*-', label='BF>%s' % bft)
            plt.legend()
            plt.ylabel(ylabel)
            plt.xlabel('N (number of subjects)')

    plt.axis('tight')
