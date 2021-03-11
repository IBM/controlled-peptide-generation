import codecs
import numpy as np
import torch
import os
import pprint
import argparse
import json

# Setup logging env
import logging

# Repo imports
import cfg

from numpy.linalg import norm
from models.model import RNN_VAE

pp = pprint.PrettyPrinter(indent=2, depth=1)

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)


class Vocab:
    '''
    Wrapper for ix2word and word2ix for converting sequences
    '''

    def __init__(self, VOCAB_PATH):
        self.fix_length = cfg.max_seq_len
        self.ix2word = {}
        self.word2ix = {}
        with codecs.open(VOCAB_PATH, 'r', 'utf-8') as f:
            for line in f:
                lsp = line.split()
                word = " ".join(lsp[:-1])
                ix = lsp[-1]
                self.ix2word[int(ix)] = word
                self.word2ix[word] = int(ix)
        LOG.info("Loaded Vocabulary.")
        self.special_tokens = set(['<unk>', '<pad>', '<start>', '<eos>'])
        self.special_tokens_ix = {self.word2ix[w] for w in self.special_tokens}

    def to_ix(self, seq, fix_length=True):
        if type(seq) == str:
            seq = seq.split()
        elif type(seq) == list:
            seq = seq
        else:
            raise ValueError('Only strings or lists of strings accepted.')
        # Make sure to have BOS and EOS symbols
        if seq[0] != "<start>":
            seq = ["<start>"] + seq
        if seq[-1] != "<eos>":
            seq = seq + ["<eos>"]
        # optionally pad seq to fix_length
        if fix_length:
            num_pads = self.fix_length - len(seq)
            seq = seq + ["<pad>"] * num_pads

        seq_ix = [self.word2ix[tok] for tok in seq]
        seq_ix = torch.LongTensor(seq_ix).view(1, -1)
        return seq_ix

    def to_word(self, seq, print_special_tokens=True):
        seq = [s.item() for s in seq]
        if not print_special_tokens:
            seq = [i for i in seq if not i in self.special_tokens_ix]
        return [self.ix2word[s] for s in seq]

    def size(self):
        return len(self.ix2word)


def load_trained_model(MODEL_PATH, n_vocab):
    '''
    Loads a pretrained model from disk.
    params:
        MODEL_PATH: location of parameter file
        n_vocab: vocabulary size
    output:
        model: model object
    '''
    model = RNN_VAE(n_vocab,
                    max_seq_len=cfg.max_seq_len,
                    **cfg.model)
    # missing_keys, _ = model.load_state_dict(torch.load(MODEL_PATH, # return values pytorch 1.1.0
    model.load_state_dict(torch.load(MODEL_PATH,
                                     map_location=lambda storage,
                                                         loc: storage),
                          strict=False)
    # assert not missing_keys, 'strict=False: only meaning to ignore AAE discriminator from AAE'
    model.device = torch.device('cpu')
    model.eval()
    return model


def encode_sequence(model,
                    vocab,
                    sequence,
                    sample_q='max'):
    '''
    encode a single (string) sequence to z.
    '''
    enc_inputs = vocab.to_ix(sequence)
    mu, logvar = model.forward_encoder(enc_inputs)
    if sample_q == 'max':
        z = mu
    else:
        z = [model.sample_z(mu, logvar) for _ in range(sample_q)]
        z = torch.cat(z, dim=0)
    return z


def sample_from_model(model,
                      vocab,
                      z=None,
                      c=None,
                      n_samples=2,
                      print_special_tokens=True,
                      **sample_kwargs):
    '''
    Wrapper for the generate_sentence function of the model
    params:
        model: model object
        z: latent space (will be sampled if not specified)
            hid_size x num_samples
        c: condition (will also be sampled if not specified)
            1 x num_samples
        sample_mode: how to generate
    '''
    # vocab_itos = vocab[0] # itos, stoi -> only need itos

    samples, z, c = model.generate_sentences(
        n_samples, z=z, c=c, **sample_kwargs)

    if sample_kwargs['sample_mode'] == 'beam':
        predictions = [[vocab.to_word(s_topK, print_special_tokens)
                        for s_topK in s] for s in samples]
    else:
        predictions = [[vocab.to_word(s, print_special_tokens)] for s in samples]

    payload = {'predictions': predictions,
               'z': z,
               'c': c}
    return payload


def interpolate_z(z_start,
                  z_end,
                  c=None,
                  method='linear',
                  n_samples=2):
    '''
    Function to generate a batch of interpolated z's between two points
    '''
    # Construct list of z's
    z_start = z_start.numpy()
    z_end = z_end.numpy()
    z_list = [z_start]

    # Compute interpolation weights
    weights = []
    if method == 'linear':
        weights = [1 / (n_samples + 1) * i for i in range(1, n_samples + 1)]
        # Generate interpolated z values
        for w in weights:
            z_list.append((1 - w) * z_start + w * z_end)
    elif method == 'tanh':
        # Compute Steps
        weights = np.array([1. / (n_samples + 1) * i
                            for i in range(1, n_samples + 1)])
        # Scale to [-2, 2] and apply tanh
        weights = np.tanh(weights * 4 - 2)
        # Scale back to [0,1]
        weights = (weights + 1) / 2
        # Generate interpolated z values
        for w in weights:
            z_list.append((1 - w) * z_start + w * z_end)
    elif method == 'slerp':
        p0 = z_start.squeeze(0)
        p1 = z_end.squeeze(0)

        def slerp(t, omega, so):
            return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

        weights = [1 / (n_samples + 1) * i for i in range(1, n_samples + 1)]
        omega = np.arccos(np.dot(p0 / norm(p0), p1 / norm(p1)))
        so = np.sin(omega)
        for w in weights:
            z_list.append(np.expand_dims(slerp(w, omega, so), 0))
    else:
        raise ValueError("Please use another interpolation method.")

    # Finish list with z_end and construct matrix
    z_list.append(z_end)
    # Construct a single matrix of z's
    z_list = np.vstack(z_list)
    # Fix first and last number in weights
    weights = list(np.concatenate(([0.], weights, [1.])))

    return z_list, weights


def generate_interpolated_samples(model,
                                  vocab,
                                  z_start,
                                  z_end,
                                  c=None,
                                  interpolation_method='linear',
                                  interpolation_samples=2,
                                  **sample_kwargs):
    '''
    Function to generate interpolated samples from a model.
    Generates samples between the two points z_start and z_end.
    '''
    z_list, weights = interpolate_z(
        z_start,
        z_end,
        c=c,
        method=interpolation_method,
        n_samples=interpolation_samples)
    # For now, just set c to 1 for every sample
    if c is None:
        c = torch.zeros((z_list.shape[0], 2))
        c[:, 1].fill_(1)

    samples = sample_from_model(model,
                                vocab,
                                z=torch.Tensor(z_list),
                                c=c,
                                n_samples=z_list.shape[0],
                                **sample_kwargs)
    samples['interpolation'] = weights
    return samples


def recon_sequence(model,
                   vocab,
                   sequence,
                   sample_q,
                   c,
                   **mb_sample_kwargs):
    """
    Reconstruct a sequence
    """
    z = encode_sequence(model, vocab, sequence, sample_q)
    # 1 (max) or multiple z samples
    n_samples = z.shape[0]
    samples = sample_from_model(
        model, vocab, z, c, n_samples, **mb_sample_kwargs)
    return samples


def interpolate_peptides(model,
                         vocab,
                         sequence_start,
                         sequence_end,
                         interpolation_kwargs={},
                         mb_sample_kwargs={}):
    with torch.no_grad():
        z_start = encode_sequence(model, vocab, sequence_start, sample_q='max')
        z_end = encode_sequence(model, vocab, sequence_end, sample_q='max')

    samples = generate_interpolated_samples(model,
                                            vocab,
                                            z_start,
                                            z_end,
                                            **interpolation_kwargs,
                                            **mb_sample_kwargs)
    return samples


def pretty_print_samples(samples, print_all_hypotheses=True):
    res = []
    for i, sample in enumerate(samples):
        if len(sample) > 1 and not print_all_hypotheses:
            sample = sample[:1]
        if len(sample) == 1:
            res.append('i {}: {}'.format(i, ' '.join(sample[0])))
        else:
            for j, hyp in enumerate(sample):
                res.append('i {} - hyp {}: {}'.format(i, j, ' '.join(hyp)))
    return '\n'.join(res)


def get_model_and_vocab_path():
    base = cfg.savepath
    # load final vae checkpoint. ignores phase 2 for now.
    MODEL_PATH = '{}/model_{}.pt'.format(base, cfg.vae.n_iter)
    # Check that model exists
    model_files = os.listdir(base)
    if MODEL_PATH.split("/")[-1] not in model_files:
        LOG.info("Selected model folder does not have fully trained model!")
        highest = max([name.split("_")[1].split(".")[0]
                       for name in model_files if "model" in name])
        LOG.info("Using iteration {} instead".format(highest))
        MODEL_PATH = '{}/model_{}.pt'.format(base, highest)
    VOCAB_PATH = '{}/vocab.dict'.format(base)
    LOG.info('api.main() load up from rundir={} model={}'.format(
        base, MODEL_PATH))
    return MODEL_PATH, VOCAB_PATH, base


def get_result_for_model(model_path, print_results=False):
    """
    Small wrapper that parses the result json file for a model
    """
    folder_name = os.path.dirname(model_path)

    # Load all results
    option_file = os.path.join(folder_name, 'result.json')
    with open(option_file, 'r') as f:
        data = json.load(f)

    # Reduce to only the model iteration
    model_name = os.path.basename(model_path)
    iteration = model_name.split(".")[0].split("_")[1]

    model_stats = {}
    for res in data:
        if str(res['it']) == str(iteration):
            model_stats = res
    if not model_stats:
        LOG.info("No results for {} found.".format(model_path))

    if print_results:
        print("Results for model {}".format(model_path))
        print(json.dumps(res, indent=2))

    return model_stats


def main(args={}):
    MODEL_PATH, VOCAB_PATH, _ = get_model_and_vocab_path()
    # Logic
    vocab = Vocab(VOCAB_PATH)
    load_trained_model(MODEL_PATH,
                       vocab.size())
    LOG.info('loaded successfully. For more tests, run evals/static_eval.py')


if __name__ == "__main__":
    LOG.info("Running API test.")
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description='Override config float & string values')
    cfg._cfg_import_export(parser, cfg, mode='fill_parser')
    parser.add_argument(
        '--seqs',
        default='''M T G E I D T A M L I G G I E F F L K
                   F A I Y Y F H E R A W Q L I R, M D K L
                   I V L K M L N S K L P Y G Q R K P F S L R''',
        help='comma separated list of seqs to reconstruct between')
    args = parser.parse_args()
    cfg._override_config(args, cfg)
    cfg._update_cfg()
    main(args)
