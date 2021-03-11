import numpy as np
import torch
import os
import pprint
import argparse

# Setup logging env
import logging

# Repo imports
import cfg

from api import (load_trained_model,
                 Vocab,
                 generate_interpolated_samples,
                 interpolate_peptides,
                 recon_sequence,
                 sample_from_model,
                 pretty_print_samples,
                 get_model_and_vocab_path,
                 get_result_for_model)

pp = pprint.PrettyPrinter(indent=2, depth=1)

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)


def test_interpolated_peptides(model, vocab):
    '''
    Test whether the model can interpolate between two fixed peptides
    with different methods
    '''
    for interpolation_method in ['linear', 'tanh', 'slerp']:
        LOG.info("INTERPOLATING WITH {} METHOD".format(interpolation_method))
        interpolation_kwargs = {'c': None,
                                'interpolation_method': interpolation_method,
                                'interpolation_samples': 9}
        mb_sample_kwargs = {'sample_mode': 'greedy'}
        peps = interpolate_peptides(model,
                                    vocab,
                                    'M L L L L L A L A L L A L L L A L L L',
                                    'M S S S S S L A A A L L',
                                    interpolation_kwargs=interpolation_kwargs,
                                    mb_sample_kwargs=mb_sample_kwargs)
        for w, p in zip(peps['interpolation'], peps['predictions']):
            print("{:.2f}".format(w), " ".join(p[0]))


def test_interpolated_z(model, vocab):
    '''
    Test whether the model can interpolate between two sampled z's
    with different methods
    '''
    z_start = model.sample_z_prior(1)
    z_end = model.sample_z_prior(1)
    c_fix = None
    print('# interpolate between z1, z2 sampled from prior. vary sampling')
    for kwargs in [{'sample_mode': 'greedy'},
                   # {'sample_mode': 'categorical', 'temp': 1.0},
                   # {'sample_mode': 'categorical', 'temp': 0.3},
                   {'sample_mode': 'beam', 'beam_size': 5, 'n_best': 3}]:

        print('### interpolate z1 z2 from prior: ', kwargs)
        samples = generate_interpolated_samples(
            model,
            vocab,
            z_start,
            z_end,
            c=c_fix,
            interpolation_method='tanh',
            interpolation_samples=11,
            **kwargs)
        for w, p in zip(samples['interpolation'], samples['predictions']):
            print("prior_zs - {:6s} - w={:.2f} - {}".format(
                kwargs['sample_mode'], w, " ".join(p[0])))


def test_sampling(model, vocab, n_samples=4):
    '''
    Test the different sampling modes
    '''
    z_fix = model.sample_z_prior(n_samples)
    c_fix = model.sample_c_prior(n_samples)
    print('# sampled z from prior, varying sample_mode')
    for kwargs in [{'sample_mode': 'greedy'},
                   {'sample_mode': 'categorical', 'temp': 1.0},
                   {'sample_mode': 'categorical', 'temp': 0.3},
                   {'sample_mode': 'beam', 'beam_size': 5, 'n_best': 3}]:
        payload = sample_from_model(
            model,
            vocab,
            z=z_fix,
            c=c_fix,
            n_samples=n_samples,
            **kwargs)
        print('### prior: ', kwargs)
        print(pretty_print_samples(payload['predictions']))


def test_reconstruction(model, vocab):
    seqs = [s.strip().split() for s in args.seqs.split(',')]
    for seq in seqs:
        print('#### reco of', ' '.join(seq), '  -- z = mu = max_z q(z|x) ')
        for mb_sample_kwargs in [{'sample_mode': 'greedy'},
                                 {'sample_mode': 'categorical', 'temp': 1.0},
                                 {'sample_mode': 'categorical', 'temp': 0.3},
                                 {'sample_mode': 'beam', 'beam_size': 5, 'n_best': 3}]:
            recos = recon_sequence(
                model, vocab, seq,
                sample_q='max', c=None,
                **mb_sample_kwargs)
            print(pretty_print_samples(recos['predictions'],
                                       print_all_hypotheses=False),
                  mb_sample_kwargs['sample_mode'])
        print('#### reco  of', ' '.join(seq), '  -- beam 15, z = 4x sampled q(z|x) ')
        mb_sample_kwargs = {'sample_mode': 'beam',
                            'beam_size': 15,
                            'n_best': 3}
        recos = recon_sequence(model,
                               vocab,
                               seq,
                               sample_q=4,
                               c=None,
                               **mb_sample_kwargs)
        print(pretty_print_samples(recos['predictions'],
                                   print_all_hypotheses=False))


def test_reconstruction_interpol(model, vocab):
    seqs = [s.strip().split() for s in args.seqs.split(',')]
    # interpolate btwn consecutive
    for seq1, seq2 in zip(seqs[:-1], seqs[1:]):
        print('#### reco interpol start source: ',
              ' '.join(seq1),
              '  -- z = mu = max_z q(z|x), beam 15')
        interpolation_kwargs = {'c': None,
                                'interpolation_method': 'tanh',
                                'interpolation_samples': 9}
        mb_sample_kwargs = {'sample_mode': 'beam',
                            'beam_size': 15,
                            'n_best': 3}
        samples = interpolate_peptides(
            model, vocab, seq1, seq2,
            interpolation_kwargs=interpolation_kwargs,
            mb_sample_kwargs=mb_sample_kwargs)
        for w, p in zip(samples['interpolation'], samples['predictions']):
            print("recon interpol - w={:.2f} - {}".format(w, " ".join(p[0])))
        print('#### reco interpol end source:   ', ' '.join(seq2))


def main(args={}):
    MODEL_PATH, VOCAB_PATH, base = get_model_and_vocab_path()
    # Logic
    vocab = Vocab(VOCAB_PATH)
    model = load_trained_model(MODEL_PATH,
                               vocab.size())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    get_result_for_model(MODEL_PATH,
                         print_results=True)

    '''
    Extract from training data
    '''
    if args.long:
        fnames = {split: "states_{}_{}.h5".format(split, cfg.vae.n_iter)
                  for split in ['train', 'val', 'test']}
        fnames = {k: os.path.join(base, fname) for k, fname in fnames.items()}
        for k, v in fnames.items():
            LOG.info("Analyzing {} at {}".format(k, v))
        if not all(os.path.exists(fname) for fname in fnames.values()):
            LOG.info("Extracting states.")
            from vis.scripts import build_index
            build_index.extract_from_dataset(
                model, vocab, cfg, base, cfg.vae.n_iter,
                max_examples=10000)
            # Not needed right now
            # build_index.build_faiss(
            #     base, cfg.vae.n_iter)
        else:
            LOG.info("States have already been extracted.")

        import matplotlib
        matplotlib.use('agg')
        from vis.scripts import (covar, kde, tsne)
        attributes = cfg.amp.attributes

        tsne.eval(fnames, attributes)
        # kde.eval(fnames, attributes)
        # covar.eval(fnames)
    '''
    Test various aspects of the model
    '''
    test_interpolated_peptides(model, vocab)
    test_sampling(model, vocab, n_samples=4)
    test_interpolated_z(model, vocab)
    test_reconstruction(model, vocab)
    test_reconstruction_interpol(model, vocab)


if __name__ == "__main__":
    LOG.info("Running API test.")
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description='Override config float & string values')
    '''
    The following arguments are optional extensions that require time to run
    KDE: estimates the density
    '''
    cfg._cfg_import_export(parser, cfg, mode='fill_parser')
    parser.add_argument(
        '--seqs',
        default='''M T G E I D T A M L I G G I E F F L K
                   F A I Y Y F H E R A W Q L I R, M D K L
                   I V L K M L N S K L P Y G Q R K P F S L R''',
        help='comma separated list of seqs to reconstruct between')
    parser.add_argument(
        '--long',
        '-long',
        action='store_true',
        default=False,
        help="""Compute the T-Sne embeddings for the data,
                colored by neg/pos/unl.""")
    args = parser.parse_args()
    cfg._override_config(args, cfg)
    cfg._update_cfg()
    main(args)
