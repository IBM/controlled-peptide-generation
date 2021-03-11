import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch

from tqdm import tqdm

# Setup logging env
import logging

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)
plt.rc('text', usetex=False)

'''
Covariance Plots
https://arxiv.org/abs/1711.00848
Variational Inference of Disentangled Latent Concepts
from Unlabeled Observations

We compute first order and second order moments of
(1) Q_z^u - marginal posterior over unlabeled.
(2) Q_z^+ - marginal posterior over amp positive.
To match the prior, Q_z mean and covar should be 0, I
'''


def eval(fnames):
    fname = fnames['train']  # TODO use fnames: train/val
    f = h5py.File(fname, 'r')
    build_covar(f, fname)


def build_covar(f, fname):
    max_evaluated = 500
    unl_mu, unl_logvar = get_enc(f, 2, max_evaluated)
    pos_mu, pos_logvar = get_enc(f, 1, max_evaluated)

    C_pos, d1_pos, d2_pos = cov_q(pos_mu, pos_logvar)
    C_unl, d1_unl, d2_unl = cov_q(unl_mu, unl_logvar)

    frob_dist_pos = analyze_one_set(C_pos, d1_pos, d2_pos, "pos", fname)
    frob_dist_unl = analyze_one_set(C_unl, d1_unl, d2_unl, "unl", fname)

    with open(fname[:-3] + "_frob_dist.txt", 'w') as g:
        g.write("Frobenius from identity for positive: {}\n".format(
            frob_dist_pos))
        g.write("Frobenius from identity for unlabeled: {}\n".format(
            frob_dist_unl))


def analyze_one_set(C, d1, d2, label_type, fname):
    '''
    Covariance for marginal posterior over amp positive
    '''
    plt.figure(figsize=(10, 10))
    plt.matshow(C.clamp_max(3), fignum=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(r'Cov$_{q_\phi}(z)$ for ' + label_type,
              pad=18, fontsize=18)
    sns.despine()
    plt.xticks(np.arange(0, C.shape[0] + 1, 20))
    plt.yticks(np.arange(0, C.shape[1] + 1, 20))
    plt.savefig(fname[:-3] + "_" + label_type + "_q_phi_z.png",
                dpi=300, format="png")

    '''
    Diagonal of the covariance
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(d1.numpy(), label='diag $\mathbb{E}_p\ \sigma$ ')
    plt.plot(d2.numpy(), label='diag $Cov_p\ \mu$')
    plt.plot(C.mean(0).numpy(), label='means: $\mathbb{E}_q\ z[z]$')
    plt.legend()
    sns.despine()
    plt.title('Diagonal of covariance for {}'.format(
        label_type), fontsize=18)
    plt.savefig(fname[:-3] + "_" + label_type + "_covar_diag.png",
                dpi=300, format="png")

    '''
    Off-diagonals
    '''
    plt.figure(figsize=(10, 5))
    offdia = C[torch.triu(torch.ones(100, 100)) == 1]
    plt.hist(offdia, bins=100)
    plt.title('Histogram of off-diagonals for {}'.format(
        label_type), fontsize=18)
    sns.despine()
    plt.savefig(fname[:-3] + "_" + label_type + "_covar_offdiag.png",
                dpi=300, format="png")

    frob_to_identity = ((C - torch.eye(100)) ** 2).sum().item()
    LOG.info("Frobenius distance to identity for {}: {}.".format(
        label_type, frob_to_identity))

    return frob_to_identity


def get_enc(f, target, max_eval=500):
    mus = []
    logvars = []
    LOG.info("Extracting {} points with label {}.".format(
        max_eval, target))
    pbar = tqdm(total=max_eval)
    num_extracted = 0
    for lab, mu, logvar in zip(f['label'], f['mu'], f['logvar']):
        if lab == target:
            mus.append(mu)
            logvars.append(logvar)
            pbar.update(1)
            num_extracted += 1
        if num_extracted >= max_eval:
            break
    pbar.close()
    mus = torch.FloatTensor(np.stack(mus, axis=0))
    logvars = torch.FloatTensor(np.stack(logvars, axis=0))
    return mus, logvars


def empirical_covar(X):
    Xcent = X - X.mean(0, keepdim=True)
    cov = (Xcent.t() @ Xcent) / Xcent.size(0)
    return cov


def cov_q(mus, logvars):
    Ep_Covq = torch.diagflat(logvars.exp().mean(0))
    Covp_Eq = empirical_covar(mus)
    return ((Ep_Covq + Covp_Eq),
            torch.diagonal(Ep_Covq),
            torch.diagonal(Covp_Eq))


def sorted_cov_diagonals_np(mus, logvars, sortix=None):
    res = sorted_cov_diagonals(torch.FloatTensor(mus),
                               torch.FloatTensor(logvars), sortix)
    return res[0].data.cpu().numpy(), res[1].data.cpu().numpy(), res[2]


def sorted_cov_diagonals(mus, logvars, sortix=None):
    """ return mean encoder variance E_p[Sigma] and activity Cov_p[mu] and sort order
    if sortix is given, this order will be used, otherwise sorted in descending activity
    """
    _, Ep_encvar, activity = cov_q(mus, logvars)
    if sortix is None:
        _, sortix = torch.sort(activity, descending=True)
    return Ep_encvar[sortix], activity[sortix], sortix


def gaussian_q_z(mus, logvars):
    mu = mus.mean(0).cpu().double().numpy()
    cov, _, _ = cov_q(mus, logvars)
    cov = cov.double().numpy()
    return scipy.stats.multivariate_normal(mu, cov)