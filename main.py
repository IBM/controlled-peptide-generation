import logging
from os.path import join as pjoin
import argparse
import random
import torch
import numpy as np

from data_processing.dataset import AttributeDataLoader
from models.model import RNN_VAE
from train_vae import train_vae
import tb_json_logger

import utils
import cfg

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.propagate = False  # do not propagate logs to previously defined root logger (if any).
formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
# console
consH = logging.StreamHandler()
consH.setFormatter(formatter)
consH.setLevel(logging.INFO)
logger.addHandler(consH)
# file handler
request_file_handler = True
log = logger

# setting up cfg
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                 description='Override config float & string values')
cfg._cfg_import_export(parser, cfg, mode='fill_parser')
cfg._override_config(parser.parse_args(), cfg)
cfg._update_cfg()
cfg._print(cfg)
cfg._save_config(parser.parse_args(), cfg, cfg.savepath)

# torch-related setup from cfg.
device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.ignore_gpu else "cpu")
log.info(f'Using device: {device}')

cfg.seed = cfg.seed if cfg.seed else random.randint(1, 10000)
log.info('Random seed: {}'.format(cfg.seed))
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

result_json = pjoin(cfg.savepath, 'result.json') if cfg.resume_result_json else None
tb_json_logger.configure(cfg.tbpath, result_json)

# DATA
dataset = AttributeDataLoader(mbsize=cfg.vae.batch_size, max_seq_len=cfg.max_seq_len,
                              device=device,
                              attributes=cfg.attributes,
                              **cfg.data_kwargs)
dataset.print_stats()
utils.save_vocab(dataset.TEXT.vocab, cfg.vocab_path)

# MODEL
if cfg.model.pretrained_emb:
    cfg.model.pretrained_emb = dataset.get_vocab_vectors()

model = RNN_VAE(n_vocab=dataset.n_vocab, max_seq_len=cfg.max_seq_len,
                **cfg.model).to(device)
log.info(model)

if cfg.loadpath:
    model.load_state_dict(torch.load(cfg.loadpath))
    log.info('Loaded model from ' + cfg.loadpath)

# ---------------------------------------------#
# Base VAE phase
# ---------------------------------------------#
if cfg.phase in [1]:
    train_vae(cfg.vae, model, dataset)

    log.info("Evaluating base vae...")

    with torch.no_grad():
        samples, _, _ = model.generate_sentences(cfg.evals.sample_size, sample_mode='categorical')
    utils.write_gen_samples(dataset.idx2sentences(samples, False), cfg.vae.gen_samples_path)

log.info(f"saving result.json and vae_result.json at {cfg.savepath}")
tb_json_logger.export_to_json(pjoin(cfg.savepath, 'result.json'))
tb_json_logger.export_to_json(pjoin(cfg.savepath, 'vae_result.json'),
                              it_filter=lambda k, v: k <= cfg.vae.n_iter)
