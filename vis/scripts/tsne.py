from collections import defaultdict
import h5py
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
import os
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Setup logging env
import logging

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)


def eval(fnames, label_dict={}):
    fname = fnames['train']
    f = h5py.File(fname, 'r')
    LOG.info("Starting with TSNE now.")
    build_tsne(f, fname, label_dict)
    LOG.info("Running latent discriminator now.")
    f_val = h5py.File(fnames['val'], 'r')
    f_test = h5py.File(fnames['test'], 'r')
    build_latent_discriminator(f, fname,
                               label_dict=label_dict,
                               val_data=f_val,
                               test_data=f_test)


def build_tsne(f, fname, label_dict={}):
    fshape = f['z'].shape
    LOG.info("Running T-SNE on {} examples of size {}.".format(
        fshape[0], fshape[1]))
    tsne = compute_tsne_embeddings(f)
    # tsne = compute_umap_embeddings(f)
    """
    Set up colors and labels
    """
    color_dict = {2: '#000000',
                  0: '#FF6859',
                  1: '#1EB980'}
    # label_list = ['neg', 'pos', 'unl']
    all_labels = f["label"][:]
    '''
    Reformat labeled and unlabeled points
    '''
    for attr_ix, (attr_name, value_key) in enumerate(label_dict):
        LOG.info("Saving {}".format(attr_name))
        legend = defaultdict(str)
        for label_name, label_int in value_key.items():
            if legend[label_int]:
                legend[label_int] += "/"
            legend[label_int] += label_name
        # Use label key for separating data
        data_points = defaultdict(list)
        for i in range(len(tsne)):
            data_points[legend[all_labels[i][attr_ix]]].append(
                tsne[i])
        # Run over data_points and plot
        plt.figure(figsize=(10, 10))
        recs = []
        legend_labs = []
        for ix, (lab, dat) in enumerate(data_points.items()):
            dat_stack = np.stack(dat)
            x = dat_stack[:, 0]
            y = dat_stack[:, 1]
            plt.scatter(x, y,
                        color=color_dict[ix],
                        alpha=.5,
                        label=lab)
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=color_dict[ix]))
            legend_labs.append(lab)
        plt.legend(recs, legend_labs)
        # Build Legend
        sns.despine()
        loc = fname[:-3] + "_" + attr_name + "_tsne.png"
        plt.savefig(loc, dpi=300, format="png")
        LOG.info("Saved T-SNE to {}".format(loc))


def save_name(h5_path, method):
    folder_name = os.path.dirname(h5_path)
    file_name = os.path.basename(h5_path).replace("states", method)
    return os.path.join(folder_name, file_name)


def save_projection(h5_path, method='umap'):
    """
    Creates a projection and saves it for later access
    """

    f = h5py.File(h5_path, 'r')

    if method == 'umap':
        emb = compute_umap_embeddings(f)
    elif method == 'tsne':
        emb = compute_tsne_embeddings(f)
    f.close()
    # Save to new h5 for caching

    h5_save = h5py.File(save_name(h5_path, method), 'w')
    h5_save.create_dataset('projection', data=emb)
    h5_save.close()

    return True


def compute_tsne_embeddings(h5_obj):
    tsne_model = TSNE(n_components=2, random_state=0)

    data = h5_obj["z"][:]
    embs = tsne_model.fit_transform(data)

    return embs


def compute_umap_embeddings(h5_obj):
    from umap import UMAP
    umap_model = UMAP(verbose=1,
                      n_neighbors=50,
                      min_dist=0.5,
                      metric='cosine')
    data = h5_obj["z"][:]
    embs = umap_model.fit_transform(data)
    return embs


def latent_disc_fn(h5fn):
    return h5fn[:-3] + "_disc.json"


def build_latent_discriminator(train_data, save_fn,
                               label_dict={},
                               val_data=None,
                               test_data=None):
    # TODO: val, test results
    results = {}
    for attr_ix, (attr_name, value_key) in enumerate(label_dict):
        LOG.info("Running Discriminators for {}".format(attr_name))
        # General setup
        X = {}
        Y_raw = {}

        X['train'] = train_data['z'][:]
        Y_raw['train'] = train_data['label'][:, attr_ix]

        if val_data is not None:
            X['val'] = val_data['z'][:]
            Y_raw['val'] = val_data['label'][:, attr_ix]
        if test_data is not None:
            X['test'] = test_data['z'][:]
            Y_raw['test'] = test_data['label'][:, attr_ix]

        # Lab vs. unlab
        Y_lab = [1 if (l >= 0) else 0 for l in Y_raw['train']]
        if max(Y_lab) > 0:
            model = LogisticRegression(solver='lbfgs', max_iter=200)
            model.fit(X['train'], Y_lab)

            for set_type in X.keys():
                cur_labs = [1 if (l >= 0) else 0 for l in Y_raw[set_type]]
                if max(cur_labs) == 0:
                    LOG.info("No labeled data for {}".format(set_type))
                    results["{}_lab_{}".format(attr_name, set_type)] = -1
                    continue
                yhat = model.predict_proba(X[set_type])

                fpr, tpr, thresholds = metrics.roc_curve(cur_labs,
                                                         yhat[:, 1],
                                                         pos_label=1)
                lab_auc = metrics.auc(fpr, tpr)
                LOG.info("lab v unlab ({}): {:.2f} AUC".format(
                    set_type,
                    lab_auc * 100))
                results["{}_lab_{}".format(attr_name, set_type)] = lab_auc

            # Between labels
            X_between = {}
            Y_between = {}
            for set_type in X.keys():
                cur_X = []
                cur_Y = []
                for x, y in zip(X[set_type], Y_raw[set_type]):
                    if y > -1:
                        cur_X.append(x)
                        cur_Y.append(y)
                X_between[set_type] = cur_X
                Y_between[set_type] = cur_Y

            if len(np.unique(Y_between['train'])) < 2:
                LOG.info(
                    "Only one label class, skipping between-label discriminator.")
                continue
        else:  # if only one label in "train"
            for set_type in X.keys():
                results["{}_lab_{}".format(attr_name, set_type)] = -1

        model = LogisticRegression(solver='lbfgs', max_iter=200)
        model.fit(X_between['train'], Y_between['train'])

        for set_type in X.keys():
            yhat = model.predict(X_between[set_type])
            between_acc = metrics.accuracy_score(Y_between[set_type], yhat)
            LOG.info("between labels ({}): {:.2f} acc".format(
                set_type,
                between_acc * 100))

            results["{}_between_{}".format(attr_name, set_type)] = lab_auc

    # Save report
    with open(latent_disc_fn(save_fn), 'w') as g:
        g.write(json.dumps(results, indent=2))

    LOG.info("Saved discriminator information to {}.".format(
        latent_disc_fn(save_fn)))
    return results
