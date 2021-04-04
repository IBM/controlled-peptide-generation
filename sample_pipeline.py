import torch

import os
import sys
import pprint
import argparse
from collections import OrderedDict
import datetime
import json
import h5py

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from modlamp.analysis import GlobalAnalysis

import cfg
from data_processing.dataset import AttributeDataLoader
from evals import PeptideEvaluator
from density_modeling import mogQ, evaluate_nll

from api import (load_trained_model,
                 Vocab,
                 get_model_and_vocab_path,
                 get_result_for_model)

import logging

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)
pp = pprint.PrettyPrinter(indent=2, depth=1)

Q_CLASS = mogQ
Q_KWARGS = {'n_components': None,
            'z_num_samples': 10,
            'covariance_type': None}


def get_encodings(query, split, model=None, dataloader=None):
    if model and dataloader:
        return get_encodings_from_dataloader(query, split, model, dataloader)
    else:
        return get_encodings_from_states(query, split)


def get_encodings_from_dataloader(query, split, model, dataloader):
    assert query == {
        'amp': 1}, 'only support this right now, else needs to convert amp=1 to amp=amp_posc posnc / and deal with ' \
                   'weighted (up)sampling between unlab/lab '
    iteratorspecs = {
        'get_encoding': {
            'subset': ['split=' + split, 'amp=amp_posc,amp_posnc'],
            'repeat': False
        }
    }
    iterators, _ = dataloader.dataset.get_subset_iterators(iteratorspecs, cfg.vae.batch_size, torch.device('cpu'))
    iterator = iterators['get_encoding']
    LOG.info('Start encoding {} samples from dataset'.format(len(iterator.dataset)))
    mus, logvars = [], []
    for batch in iter(iterator):
        with torch.no_grad():
            (mu, logvar), (z, c), dec_logits = model(
                batch.text, q_c='classifier', sample_z='max')
            mus.append(mu.detach())
            logvars.append(logvar.detach())
    mus, logvars = torch.cat(mus, dim=0), torch.cat(logvars, dim=0)
    return mus, logvars


def get_encodings_from_states(query, split):
    base = cfg.savepath
    attr_to_colix = {k: i for i, (k, _) in enumerate(cfg.attributes)}

    fnames = {split: "states_{}_{}.h5".format(split, cfg.vae.n_iter)
              for split in ['train', 'val', 'test']}
    fnames = {k: os.path.join(base, fname) for k, fname in fnames.items()}
    assert all(os.path.exists(fname) for fname in fnames.values()
               ), 'need dumped states, run static_eval first'
    fn = fnames[split]
    f = h5py.File(fn, 'r')
    mu, logvar = (torch.from_numpy(f['mu'][:]).double(),
                  torch.from_numpy(f['logvar'][:]).double())
    lab = torch.from_numpy(f['label'][:])
    N = lab.shape[0]
    sel = torch.ones(N).byte()
    for attr_name, val in query.items():
        attr_ix = attr_to_colix[attr_name]
        sel = sel & (lab[:, attr_ix] == val)
    return mu[sel], logvar[sel]


def fitQ_and_test(QClass, QKwargs, Q_select={}, negative_select={}, model=None, dataloader=None):
    """
    Fit Q_xi^a(z) based on samples with single
    "attribute=y" <- Q_select query.
    Collect metrics: Q_xi^a(z),
                     p(z)
                     nll on heldout positive and
                     heldout negative samples.
    """
    if model and dataloader:
        mu, logvar = get_encodings_from_dataloader(query=Q_select, split='train,val',
                                                   model=model, dataloader=dataloader)
    else:
        mu, logvar = get_encodings_from_states(query=Q_select, split='train')
    Q_xi_a = QClass(mu, logvar, **QKwargs)

    LOG.info('Fitted {}  {} on selection {}'.format(QClass.__name__,
                                                    str(QKwargs),
                                                    str(Q_select)))

    eval_points = [
        ('a,tr', get_encodings_from_states(split='train', query=Q_select)),
        ('a,hld', get_encodings_from_states(split='test', query=Q_select)),
        # ('!a,hld', get_encodings(split='val', query=negative_select))
    ]

    metrics = OrderedDict()
    for name, points in eval_points:
        nllq, nllp = evaluate_nll(Q_xi_a, points)
        # key = r'CE$(q^{{ {} }} |{{}})$'.format(name)
        metrics[name] = (nllq, nllp)
    return Q_xi_a, metrics


def decode_from_z(z, model, dataset):
    sall = []
    LOG.info('Decoder decoding: beam search')
    for zchunk in torch.split(z, 1024):
        s, _, _ = model.generate_sentences(zchunk.size(0),
                                           zchunk,
                                           sample_mode='beam',
                                           beam_size=5)
        s = [hypotheses[0] for hypotheses in s]
        sall += s
    return dataset.idx2sentences(sall, print_special_tokens=False)


def save_csv_pkl(samples, fn):
    outfn = fn + '.csv'
    samples.drop(columns='z').to_csv(outfn, index_label='idx')
    outfn = fn + '.pkl'
    samples.to_pickle(outfn)


def save_samples(samples, basedir, fn_prefix):
    outfn = os.path.join(basedir, fn_prefix)
    outfn += '_{}'.format(datetime.datetime.now().isoformat().split('T')[0])
    with open(outfn + '.plain.txt', 'w') as fh:
        fh.write(samples['peptide'].to_string(index=False))
    save_csv_pkl(samples, outfn)
    LOG.info('Full sample list written to {}.pkl/csv'.format(outfn))
    accepted = samples[samples.accept]
    accepted_fn = '{}.accepted.{}'.format(outfn, len(accepted))
    save_csv_pkl(accepted, accepted_fn)
    LOG.info('Accepted sample list written to {}.pkl/csv'.format(accepted_fn))


def score_clfZ(clf, z):
    z = z.numpy()
    RETURN_LABEL_COL_IX = 1
    probs = clf.predict_proba(z)[:, RETURN_LABEL_COL_IX]
    return probs


def build_clfZ(attr):
    """
    sklearn logistic reg clf between attr=1 and attr=0.
    based on vis/scripts/tsne.py

    ASSUMES that for attr we get -1, 0, 1 labels,
    corresponding to na / neg / pos
    """
    zpos_mu, zpos_logvar = get_encodings_from_states(query={attr: 1}, split='train')
    zneg_mu, zneg_logvar = get_encodings_from_states(query={attr: 0}, split='train')
    Y = torch.cat([torch.ones(zpos_mu.shape[0]),
                   torch.zeros(zneg_mu.shape[0])],
                  dim=0)
    X = torch.cat([zpos_mu, zneg_mu], dim=0)
    X, Y = X.numpy(), Y.numpy()

    clf = LogisticRegression(solver='lbfgs', max_iter=200)
    clf.fit(X, Y)
    acc = clf.score(X, Y)
    LOG.info('Fitted LogReg classifier in z-space, on attr={}.'.format(
        attr))
    LOG.info('num samples: {} pos, {} neg. train accuracy={:.5f}'.format(
        zpos_mu.shape[0], zneg_mu.shape[0], acc))
    return clf


def get_new_samples(model, dataset, Q, n_samples):
    """
    Get one round of sampled z's and decode
    """

    samples_z, scores_z, accept_z = Q.rejection_sample(n_samples=n_samples)
    samples = decode_from_z(samples_z, model, dataset)
    df = pd.DataFrame({'peptide': samples,
                       # 'sample_source': get_sample_source_str(),
                       'z': [tuple(z.tolist()) for z in samples_z],
                       'accept_z': accept_z,
                       **scores_z})
    return df


def compute_modlamp(df):
    ana_obj = GlobalAnalysis(df.peptide.str.replace(' ', ''))
    ana_obj.calc_H()
    ana_obj.calc_uH()
    ana_obj.calc_charge()
    df.loc[:, 'H'] = ana_obj.H[0]
    df.loc[:, 'uH'] = ana_obj.uH[0]
    df.loc[:, 'charge'] = ana_obj.charge[0]
    return df


def one_sampling_round(model, dataset, Q, n_samples_per_round):
    """
    Generate one round of samples
    """
    samples_df = get_new_samples(model, dataset, Q, n_samples_per_round)
    samples_df = compute_modlamp(samples_df)
    mask_accept = samples_df['accept_z']
    samples_df['accept'] = mask_accept
    return samples_df


def get_sample_source_str():
    return ' '.join(sys.argv[1:])


def main(args={}):
    MODEL_PATH, VOCAB_PATH, _ = get_model_and_vocab_path()
    LOG.info('Load model, vocab, dataloader.')
    vocab = Vocab(VOCAB_PATH)
    model = load_trained_model(MODEL_PATH,
                               vocab.size())
    LOG.info('Loaded model succesfully.')
    LOG.info('Set up dataset, evaluator objects')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = AttributeDataLoader(
        mbsize=cfg.vae.batch_size,
        max_seq_len=cfg.max_seq_len,
        device=torch.device('cpu'),
        attributes=cfg.attributes,
        **cfg.data_kwargs)
    pep_evaluator = PeptideEvaluator(
        orig_filename=os.path.join(cfg.datapath,
                                   'peptide_evals/pos_amp_reffiles',
                                   'positive_orig_l15.txt'),
        seq_len=cfg.max_seq_len)

    metrics = get_result_for_model(MODEL_PATH,
                                   print_results=False)
    LOG.info('Model metrics:')
    pp.pprint(metrics)

    LOG.info('Fit attribute-conditioned marginal posterior Q_xi^a(z)')
    for k in Q_KWARGS:
        if 'Q_' + k in dir(args):
            Q_KWARGS[k] = getattr(args, 'Q_' + k)

    if args.Q_select_amppos:
        Q_SELECT_QUERY = {'amp': 1}
        Q_NEGATIVE_QUERY = {'amp': 0}
    else:
        Q_SELECT_QUERY = {}
        Q_NEGATIVE_QUERY = {}
    Q, Q_xi_metrics = fitQ_and_test(Q_CLASS,
                                    Q_KWARGS,
                                    Q_SELECT_QUERY,
                                    Q_NEGATIVE_QUERY,
                                    model if args.Q_from_full_dataloader else None,
                                    dataset if args.Q_from_full_dataloader else None)
    LOG.info('Q Fit metrics: ')
    print(json.dumps(Q_xi_metrics, indent=4))

    z_clfs = {}
    for attr in ['amp', 'tox']:
        clf_zspace = build_clfZ(attr)
        z_clfs[attr] = clf_zspace

    Q.init_attr_classifiers(z_clfs, clf_targets={'amp': 1, 'tox': 0})

    '''
    SETUP DONE, SAMPLING BELOW
    '''

    samples = pd.DataFrame(columns=['peptide'])
    round_ix = 0

    def is_finished(df, min_accepted):
        unfinished = len(df) < min_accepted or df['accept'].sum() < min_accepted
        return not unfinished

    while not is_finished(samples, args.n_samples_acc):
        round_ix += 1
        LOG.info("Round #{}".format(round_ix))
        new_samples = one_sampling_round(
            model,
            dataset,
            Q,
            args.n_samples_per_round)

        new_samples = new_samples.loc[new_samples.peptide.drop_duplicates().index]
        new_samples = new_samples[~new_samples['peptide'].isin(
            samples['peptide'])]
        samples = pd.concat([samples, new_samples], ignore_index=True, sort=False)
        dropped_num = args.n_samples_per_round - new_samples.shape[0]
        if dropped_num > 0:
            LOG.info("Dropped {} duplicate samples".format(dropped_num))
        LOG.info('Q_xi(z|a) rejection sampling acceptance rate: {}/{} = {:.4f}'.format(
            samples['accept_z'].sum(), len(samples), 100.0 * samples['accept_z'].sum() / len(samples)))
        LOG.info('     - full filter pipeline accepted: {}/{} = {:.4f}'.format(
            samples['accept'].sum(), len(samples), 100.0 * samples['accept'].sum() / len(samples)))

    save_samples(samples, cfg.savepath, args.samples_outfn_prefix)


if __name__ == "__main__":
    LOG.info("Sample pipeline. Fit Q_xi(z), Sample from it, score samples.")
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description='Override config float & string values')
    cfg._cfg_import_export(parser, cfg, mode='fill_parser')
    parser.add_argument('--QClass', default='mogQ')
    parser.add_argument(
        '--Q_n_components', type=int, default=100,
        help='mog num components for Q model')
    parser.add_argument(
        '--Q_covariance_type', default='diag',
        help='mog Q covariance type full|tied|diag')
    parser.add_argument(
        '--n_samples_per_round', type=int, default=5000,
        help='number of samples to generate & evaluate.')
    parser.add_argument(
        '--n_samples_acc', type=int, default=100,
        help='number of samples to generate & evaluate.')
    parser.add_argument(
        '--samples_outfn_prefix', default='samples',
        help='''prefix to fn to write out the samples.
                Will have .txt .csv .pkl versions''')
    parser.add_argument(
        '--Q_select_amppos', type=int, default=0,
        help='select amp positive to fit Q_xi or not.')
    parser.add_argument(
        '--Q_from_full_dataloader', action='store_true', default=False,
        help='to fit Q_z, select from full dataloader')
    args = parser.parse_args()

    cfg._override_config(args, cfg)
    cfg._update_cfg()
    cfg._print(cfg)
    main(args)
