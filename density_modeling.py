import torch

import numpy as np
import sklearn.mixture
import scipy.stats
import math

from vis.scripts.covar import empirical_covar


def prior_logpdf(z):
    D = z.shape[0]
    energy = 0.5 * (z ** 2).sum()
    return -0.5 * D * math.log(math.tau) - energy.item()


class fullQ:
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        self.diagcovarinv = 1. / logvar.exp()
        self.N, self.D = mu.shape
        self.logdets = self.logvar.sum(1)  # det = prod(var diagonals) so logdet = trace(logvar)

    def pdf(self, x):
        assert x.ndim() == 1, 'expecting  single sample'
        return math.exp(self.logpdf(x))

    def logpdf(self, x):
        assert x.dim() == 1, 'expecting  single sample'
        x = x.view(1, self.D).double()
        energy = 0.5 * (((self.mu - x) ** 2) * self.diagcovarinv).sum(1)
        logpdf_perN = -0.5 * self.D * math.log(math.tau) - 0.5 * self.logdets - energy
        ret = torch.logsumexp(logpdf_perN, dim=0) - math.log(self.N)
        return ret.item()


class RejSampleBase:
    def init_attr_classifiers(self, attr_clfs, clf_targets):
        self.attr_clfs = attr_clfs
        self.clf_targets = clf_targets

    def score_clf(self, attr_name, z):
        z = z.numpy()
        clf = self.attr_clfs[attr_name]
        TARGET_COL_IX = self.clf_targets[attr_name]
        probs = clf.predict_proba(z)[:, TARGET_COL_IX]
        return probs

    def rejection_sample(self, n_samples, prefix='clfZ'):
        # torch Tensor n_samples x D
        samples_z = self.sample(n_samples)
        scores_z = {prefix + '_prob_accum': 1.0}
        for attr in self.attr_clfs:
            k = '{}_{}={}'.format(prefix, attr, self.clf_targets[attr])
            scores_z[k] = self.score_clf(attr, samples_z)
            scores_z[prefix + '_prob_accum'] *= scores_z[k]
        uniform_rand = np.random.uniform(size=n_samples)
        accepted = uniform_rand < scores_z[prefix + '_prob_accum']
        return samples_z, scores_z, accepted


class mogQ(RejSampleBase):
    def __init__(self, mu, logvar, n_components=10, z_num_samples=10, **mog_kwargs):
        self.mu = mu
        self.logvar = logvar
        self.N, self.D = mu.shape
        self.z = torch.cat([mu + (0.5 * logvar).exp() * torch.randn_like(logvar) for s in range(z_num_samples)], dim=0)
        self.n_components = n_components
        self.mog = sklearn.mixture.GaussianMixture(n_components=self.n_components, **mog_kwargs)
        self.mog.fit(self.z.cpu().numpy())
        print('mog-{}. Converged: {} in {} iters, log likelihood lower bound: {:.4f}'.format(
            self.n_components, self.mog.converged_, self.mog.n_iter_, self.mog.lower_bound_))

    def logpdf(self, x):
        assert x.dim() == 1, 'expecting  single sample'
        return self.mog.score(x.view(1, -1).numpy())

    def sample(self, n_samples):
        return torch.from_numpy(self.mog.sample(n_samples)[0]).float()


class gaussianQ:
    def __init__(self, mu, logvar, covar_add_encoder_vars=True):
        self.covar = empirical_covar(mu)
        if covar_add_encoder_vars:
            self.covar += torch.diagflat(logvar.exp().mean(0))
        self.mean = mu.mean(0)
        self.gaussian = scipy.stats.multivariate_normal(self.mean.numpy(), self.covar.numpy())

    def logpdf(self, x):
        assert x.dim() == 1, 'in fact scipy multivariate_normal can accept any prepending dimensions'
        return self.gaussian.logpdf(x.numpy())

    def sample(self, n_samples):
        return torch.from_numpy(self.gaussian.rvs(size=n_samples)).float()


def evaluate_nll(q, points):
    """ returns nll of points under Q(z) and p(z). """
    mu, lv = points  # tuple of N x d
    N = mu.shape[0]
    llp, llq = 0.0, 0.0
    for s in range(N):
        z = mu[s] + (0.5 * lv[s]).exp() * torch.randn(1).item()
        llq += q.logpdf(z)
        llp += prior_logpdf(z)
    return -llq / N, -llp / N
