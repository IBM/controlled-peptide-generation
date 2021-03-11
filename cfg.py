import sys, os, types
import json
from collections import OrderedDict
from utils import check_dir_exists


# small helper stuff
class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


def _override_config(args, cfg):
    """ call _cfg_import_export in override mode, update cfg from:
        (1) contents of config_json (taken from (a) loadpath if not auto, or (2) savepath)
        (2) from command line args
    """
    config_json = vars(args).get('config_json', '')
    _cfg_import_export(args, cfg, mode='override')


def _override_config_from_json(cfg, config_json):
    if config_json:
        override_vals = Bunch(json.load(open(config_json)))
    # Now actually import into cfg
    _cfg_import_export(override_vals, cfg, mode='override')


def _save_config(cfg_overrides, cfg_complete, savepath):
    json_fn = os.path.join(savepath, 'config_overrides.json')
    check_dir_exists(json_fn)
    with open(json_fn, 'w') as fh:
        json.dump(vars(cfg_overrides), fh, indent=2, sort_keys=True)
    json_fn = os.path.join(savepath, 'config_complete.json')
    with open(json_fn, 'w') as fh:
        d = {}
        _cfg_import_export(d, cfg_complete, mode='fill_dict')
        json.dump(d, fh, indent=2, sort_keys=True)
    # add if desired: _copy_to_nested_dict(cfg_complete) dump


def _copy_to_nested_dict(cfg_):
    """ follows _cfg_import_export() flow but creates nested dictionary """
    ret = {}
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            ret[k] = v
        elif type(v) == Bunch:  # recurse; descend into Bunch
            ret[k] = _copy_to_nested_dict(v)
    return ret


def _cfg_import_export(cfg_interactor, cfg_, prefix='', mode='fill_parser'):
    """ Iterate through cfg_ module/object. For known variables import/export
    from cfg_interactor (dict, argparser, or argparse namespace) """
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            if mode == 'fill_parser':
                cfg_interactor.add_argument('--{}{}'.format(prefix, k), type=type(v), help='default: {}'.format(v))
            elif mode == 'fill_dict':
                cfg_interactor['{}{}'.format(prefix, k)] = v
            elif mode == 'override':
                prek = '{}{}'.format(prefix, k)
                if prek in cfg_interactor:
                    setattr(cfg_, k, getattr(cfg_interactor, prek))
        elif type(v) == Bunch:  # recurse; descend into Bunch
            _cfg_import_export(cfg_interactor, v, prefix=prefix + k + '.', mode=mode)


def _update_cfg():
    """ function to update/postprocess based on special cfg values """
    global tiny, vae, full, partN, part, phase, resume_result_json, runname, seed, \
        savepath_toplevel, tb_toplevel, savepath, tbpath, loadpath, vocab_path, \
        dataset
    # dataset, dataset_unl, dataset_lab
    # constructing savepath and resultpath
    savepath = os.path.join(savepath_toplevel, runname)  # {savepath}/model_{iter}.pt
    tbpath = os.path.join(tb_toplevel, runname)  # {tbpath}/eventfiles

    if tiny:  # tiny data & iters for fast debugging. Use shared, will override train/full.
        shared.n_iter = 100
        shared.cheaplog_every = 10
        shared.expsvlog_every = 25
        evals.sample_size = 30
        shared.batch_size = 5
        full.s_iter = shared.n_iter
        resume_result_json = False  # for testing overwrite, minimal hassle
    if partN > 1:
        assert phase > 0, 'split in parts only makes sense when doing per-phase split'
        cfgv = vae if phase == 1 else full
        cfgv.n_iter = cfgv.n_iter // partN
        cfgv.s_iter += part * cfgv.n_iter
        cfgv.expsvlog_every = min(cfgv.expsvlog_every, cfgv.n_iter)
        assert (
                       cfgv.s_iter + cfgv.n_iter) % cfgv.expsvlog_every == 0, 'Final model wont be saved; n_iter={}, expsvlog_every {}'.format(
            cfgv.n_iter, cfgv.expsvlog_every)
    # inject shared fields into vae and full
    vae.update(shared)
    full.update(shared)
    # Vocab path
    if vocab_path == 'auto':
        vocab_path = os.path.join(savepath, 'vocab.dict')
    # checkpoint paths: inject into cfgv, and use to define auto-loadpath.
    chkpt_path = os.path.join(savepath, 'model_{}.pt')
    vae.chkpt_path = chkpt_path
    full.chkpt_path = chkpt_path
    if loadpath == 'auto':
        if part == 0 and phase != 2:  # start from scratch
            loadpath = ''
        else:  # auto load from s_iter
            cfgv = vae if phase == 1 else full
            loadpath = chkpt_path.format(cfgv.s_iter)
    # seeding
    if seed and phase > 0:  # increment the seed to have new seeds per sub-run: different loader shuffling, model/training stochasticity
        seed += (phase - 1) * partN + part

    # set result fns
    def set_result_filenames(cfgv, savepath, list_of_fns):
        for fieldname, fn in list_of_fns:
            cfgv[fieldname] = os.path.join(savepath, fn)

    set_result_filenames(vae, savepath,
                         [('gen_samples_path', 'vae_gen.txt'), ('eval_path', 'vae_eval.txt'),
                          ('fasta_gen_samples_path', 'vae_gen.fasta')])
    set_result_filenames(full, savepath,
                         [('gen_samples_path', 'full_gen.txt'), ('samez_samples_path', 'full_samez.txt'),
                          ('posz_samples_path', 'full_posz.txt'), ('interp_samples_path', 'full_interp.txt'),
                          ('eval_path', 'full_eval.txt'), ('pos_eval_path', 'full.pos_eval.txt'),
                          ('fasta_gen_samples_path', 'full_gen.fasta'), ('fasta_pos_samples_path', 'pos_gen.fasta')])
    # switch dataset
    _set_dataset(dataset)


def _print(cfg_, prefix=''):
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            print('{}{}\t{}'.format(prefix, k, v))
        elif type(v) == Bunch:  # recurse; descend into Bunch
            print('{}{}:'.format(prefix, k))
            _print(v, prefix + '  |- ')


# general
config_json = ''  # set path to load config.json. load order: 1) argparse 2) json
ignore_gpu = False
seed = 1238
tiny = False

# paths
tb_toplevel = 'tb'  # tb/run_name_with_hypers/eventfiles
savepath_toplevel = 'output'  # output/run_name_with_hypers/{checkpoints, generated sequences, etc}
runname = 'default'  # override on command line in spj
datapath = 'data'
# savepath = os.path.join(savepath_toplevel, runname) # {savepath}/model_{iter}.pt
# tbpath   = os.path.join(tb_toplevel, runname) # {tbpath}/eventfiles
loadpath = 'auto'  # autofill: savepath + right iter based in s_iter
vocab_path = 'auto'  # autofill: savepath + vocab.dict
phase = -1  # -1: both, 1: vae, 2: full
part = 0  # partN > 1 splits up s_iter, n_iter
partN = 1
resume_result_json = True  # load up and append to result.json by default

# vae - pretraining
vae = Bunch(
    batch_size=32,
    lr=1e-3,
    # TODO lrate decay with scheduler
    s_iter=0,
    n_iter=200000,
    beta=Bunch(
        start=Bunch(val=1.0, iter=0),
        end=Bunch(val=2.0, iter=10000)
    ),
    lambda_logvar_L1=0.0,  # default from https://openreview.net/pdf?id=r157GIJvz
    lambda_logvar_KL=1e-3,  # default from https://openreview.net/pdf?id=r157GIJvz
    z_regu_loss='mmdrf',  # kl (vae) | mmd (wae) | mmdrf (wae)
    cheaplog_every=500,  # cheap tensorboard logging eg training metrics
    expsvlog_every=20000,  # expensive logging: model checkpoint, heldout set evals, word emb logging
)
vae.beta.start.iter = vae.s_iter
vae.beta.end.iter = vae.s_iter + vae.n_iter // 5

# full training
full = Bunch(
    batch_size=32,
    lrE=3e-4,  # encoder
    lrG=3e-4,  # generator
    lrC=3e-4,  # classifier
    # TODO lrate decay with scheduler
    # n_iter = 10000,   # default for yelp
    n_iter=50000,  # default for AMP
    # n_iter = 5,
    s_iter=vae.n_iter,
    classifier_min_length=5,  # specific to classifier architecture
    # hypers for controlled text gen
    beta=Bunch(
        start=Bunch(val=2.0, iter=vae.n_iter),
        end=Bunch(val=2.0, iter=vae.n_iter + 50000)
    ),
    z_regu_loss='mmdrf',  # kl (vae) | mmd (wae) | mmdrf (wae)
    C_hard_sample_kwargs=Bunch(
        sample_mode='categorical',  # sample temp: annealing see above
    ),
    G_soft_sample_kwargs=Bunch(
        sample_mode='none_softmax',
        # gumbel_temp=1.0,
    ),
    softmax_temp=Bunch(
        start=Bunch(iter=vae.n_iter, val=1.0),
        end=Bunch(iter=vae.n_iter + 20000, val=1.0)
    ),
    lambda_e=0.1,  # entropy
    lambda_c=1.0,  # G: loss_attr_c
    lambda_z=0.1,  # G: loss_attr_z
    lambda_u=0.1,  # D: unsup (vs sup=1.0)
    lambda_logvar_L1=0.0,  # default from https://openreview.net/pdf?id=r157GIJvz
    lambda_logvar_KL=1e-3,  # default from https://openreview.net/pdf?id=r157GIJvz
    cheaplog_every=50,  # cheap tensorboard logging eg training metrics
    expsvlog_every=2000,  # expensive logging: model checkpoint, heldout set evals, word emb logging
)
full.beta.start.iter = full.s_iter
full.beta.end.iter = full.s_iter + full.n_iter
full.softmax_temp.start.iter = full.s_iter
full.softmax_temp.end.iter = full.s_iter + full.n_iter

# shared settings, are injected in train & full Bunch in _update_cfg()
shared = Bunch(
    clip_grad=5.0,
)

# evals settings
evals = Bunch(
    sample_size=2000,
    sample_modes=Bunch(
        # cat  = Bunch(sample_mode='categorical', temp=0.8),
        beam=Bunch(sample_mode='beam', beam_size=5, n_best=3),
    ),
)

# config for the losses, constant during training & phases
losses = Bunch(
    wae_mmd=Bunch(
        sigma=7.0,  # ~ O( sqrt(z_dim) )
        kernel='gaussian',
        # for method = rf
        rf_dim=500,
        rf_resample=False
    ),
)

max_seq_len = 25

# model architecture
model = Bunch(
    z_dim=100,
    c_dim=2,
    emb_dim=150,
    pretrained_emb=None,  # set True to load from dataset_unl.get_vocab_vectors()
    freeze_embeddings=False,
    flow=0,
    flow_type='',
    E_args=Bunch(
        h_dim=80,  # 20 for amp, 64 for yelp
        biGRU=True,
        layers=1,
        p_dropout=0.0
    ),
    G_args=Bunch(
        G_class='gru',
        GRU_args=Bunch(
            # h_dim = (z_dim + c_dim) for now. TODO parametrize this?
            p_word_dropout=0.3,
            p_out_dropout=0.3,
            skip_connetions=False,
        ),
        deconv_args=Bunch(
            max_seq_len=max_seq_len,
            num_filters=100,
            kernel_size=4,
            num_deconv_layers=3,
            useRNN=False,
            temperature=1.0,
            use_batch_norm=True,
            num_conv_layers=2,
            add_final_conv_layer=True,
        ),
    ),
    C_args=Bunch(
        min_filter_width=3,
        max_filter_width=5,
        num_filters=100,
        dropout=0.5
    )
)

# dataset
dataset = 'amp'  # amp / yelp / sst. Switch to set other stuff based on this.
data_kwargs, data_prefixes = None, None  # will be filled in by _set_dataset()


def _set_dataset(dataset):
    # dataset: amp / yelp / sst
    global amp, yelp  # bunch with specs
    global data_kwargs, data_prefixes, evals, attributes  # bunch with specs
    if dataset == 'amp':
        ds_bunch = amp
    elif dataset == 'yelp':
        ds_bunch = yelp
    else:
        raise ValueError('unknown dataset ' + dataset)
    data_kwargs = ds_bunch.data_kwargs
    data_prefixes = ds_bunch.data_prefixes
    attributes = ds_bunch.attributes


# set path to your data
DATA_ROOT = './PATH_TO_DATA/'

amp_sample_prob_factors = {
    'amp=amp_posc': 20, 'amp=amp_posnc': 10,
    'amp=amp_negc': 20, 'amp=amp_negnc': 10,
    'tox=tox_posc': 20, 'tox=tox_posnc': 10,
    'tox=tox_negc': 20, 'tox=tox_negnc': 10,
    'sol': 20,
    'anticancer': 20, 'antihyper': 20, 'hormone': 20
}

amp = Bunch(
    data_kwargs=Bunch(
        lower=False,
        data_path=DATA_ROOT + 'amp/' if not 'DATA_PATH_AMP' in os.environ else os.environ['DATA_PATH_AMP'],
        data_format='csv',
        csv_files=['unlab.csv', 'amp_lab.csv', 'tox_lab.csv', 'sol_lab.csv', \
                   'anticancer.csv', 'antihypertensive.csv', 'cell-cell.csv'],
        iteratorspecs=Bunch(
            train_vae=Bunch(subset=['split=train'], weighted_random_sample=True,
                            sample_prob_factors=amp_sample_prob_factors),
            train_amp_lab=Bunch(subset=['split=train', 'amp'], weighted_random_sample=True,
                                sample_prob_factors=amp_sample_prob_factors),
            hld_vae=Bunch(subset=['split=val'], weighted_random_sample=True,
                          sample_prob_factors=amp_sample_prob_factors),
            hld_unl=Bunch(subset=['split=val', '^amp'], ),
            hld_amppos=Bunch(subset=['split=val', 'amp=amp_posc,amp_posnc'], ),
            hld_ampneg=Bunch(subset=['split=val', 'amp=amp_negc,amp_negnc'], )
        ),
        fixed_vocab_path=DATA_ROOT + 'amp/vocab.dict',
        split_seed=1288,  # Purely for computing the train/val/test split.
    ),
    data_prefixes=Bunch(
        dataset_type='bio',
        # for smry refvals
        dataset_unl='amp_unlabeled',
        dataset_lab='amp_labeled',
    ),
    attributes=[
        ('amp', {'amp_negnc': 0, 'amp_negc': 0, 'amp_posc': 1, 'amp_posnc': 1, 'na': -1}),
        ('tox', {'tox_negc': 0, 'tox_negnc': 0, 'tox_posc': 1, 'tox_posnc': 1, 'na': -1}),
        ('sol', {'sol_neg': 0, 'sol_pos': 1, 'na': -1}),
        ('anticancer', {'anticancer': 1, 'na': -1}),
        ('antihyper', {'antihyper': 1, 'na': -1}),
        ('hormone', {'cell': 1, 'na': -1})
    ],
)

_set_dataset(dataset)  # will update cfg.data = cfg.amp
