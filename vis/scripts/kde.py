import h5py
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

from tqdm import tqdm
import seaborn as sns

# Setup logging env
import logging

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.info)


def eval(fnames):
    fname = fnames['train'] # TODO use fnames: train/val
    f = h5py.File(fname, 'r')
    max_evaluated = 500

    LOG.info('building gaussians now.')
    gaussians = []
    ix = 0
    for mu, logvar in tqdm(zip(f['mu'], f['logvar'])):
        ix += 1
        gaussians.append(norm(mean=mu, cov=to_var(logvar)))

    LOG.info('evaluating gaussians now. This takes a while...')
    # TRAIN RATIOS
    ratios_train_lab = []
    densities_train_lab = []
    ratios_train_unlab = []
    densities_train_unlab = []
    ix_lab = 0
    ix_unlab = 0
    pbar1 = tqdm(total=max_evaluated, position=1, desc="Unlabeled")
    pbar2 = tqdm(total=max_evaluated, position=0, desc="Labeled")
    for z, l in zip(f['z'], f['label']):
        if l < 2:
            ix_lab += 1
            if ix_lab > max_evaluated:
                continue
            else:
                pbar1.update(1)
            r, d = estimate_density(gaussians, z)
            ratios_train_lab.append(r)
            densities_train_lab.append(d)
        else:
            ix_unlab += 1
            if ix_unlab > max_evaluated:
                continue
            else:
                pbar2.update(1)
            r, d = estimate_density(gaussians, z)
            ratios_train_unlab.append(r)
            densities_train_unlab.append(d)
    pbar1.close()
    pbar2.close()

    LOG.info("Lab: {:.2f}% Non-zero, {} Avg density".format(
        np.mean(ratios_train_lab)*100,
        np.mean(densities_train_lab)))

    LOG.info("Unlab: {:.2f}% Non-zero, {} Avg density".format(
        np.mean(ratios_train_unlab)*100,
        np.mean(densities_train_unlab)))

    with open(fname[:-3]+"_kde.txt", 'w') as g:
        g.write("Lab: {:.2f}% Non-zero, {} Avg density\n".format(
            np.mean(ratios_train_lab)*100,
            np.mean(densities_train_lab)))
        g.write("Unlab: {:.2f}% Non-zero, {} Avg density".format(
            np.mean(ratios_train_unlab)*100,
            np.mean(densities_train_unlab)))

    plt.figure(figsize=(10, 5))
    plt.hist(ratios_train_lab, bins=40, alpha=.8, label="Labeled")
    plt.hist(ratios_train_unlab, bins=40, alpha=.8, label="Unlabeled")
    sns.despine()
    plt.title("Fraction of Gaussians with non-zero density")
    plt.legend()
    plt.savefig(fname[:-3]+"_gaussians.png", dpi=300, format="png")
    LOG.info("Saved Gaussian figure to {}".format(fname[:-3]+"_tsne.png"))


def to_var(logvar):
    return np.diag(np.sqrt(np.exp(logvar)))


def estimate_density(gaussians, x):
    densities = np.array([g.pdf(x) for g in gaussians])
    mean_density = np.mean(densities)
    return sum(densities > 0)/len(densities), mean_density