import faiss
import h5py
import logging
import numpy as np
import torch
import os

from tqdm import tqdm

LOG = logging.getLogger('GenerationAPI')
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)


def extend_set(setname, bsize):
    # Helper function that increases h5py set size
    setname.resize(setname.shape[0] + bsize, axis=0)


def extend_sets(setnames, bsize):
    for setname in setnames:
        extend_set(setname, bsize)


def resize_sets(setnames, size):
    for setname in setnames:
        setname.resize(size, axis=0)


def setup_hdf5_file(fn, cfg):
    """ sets up datasets, returns open descriptor """
    f = h5py.File(fn, "a")
    srcset = f.create_dataset("src",
                              (0,
                               cfg.max_seq_len),
                              maxshape=(None, None),
                              dtype="int",
                              chunks=(10, 5),
                              compression="gzip",
                              compression_opts=9)
    zset = f.create_dataset("z",
                            (0,
                             cfg.model.z_dim),
                            maxshape=(None, None),
                            dtype="float16",
                            chunks=(10, 5),
                            compression="gzip",
                            compression_opts=9)
    muset = f.create_dataset("mu",
                             (0,
                              cfg.model.z_dim),
                             maxshape=(None, None),
                             dtype="float16",
                             chunks=(10, 5),
                             compression="gzip",
                             compression_opts=9)
    logvarset = f.create_dataset("logvar",
                                 (0,
                                  cfg.model.z_dim),
                                 maxshape=(None, None),
                                 dtype="float16",
                                 chunks=(10, 5),
                                 compression="gzip",
                                 compression_opts=9)
    labelset = f.create_dataset("label",
                                (0, len(cfg.attributes)),
                                maxshape=(None, None),
                                dtype="int",
                                chunks=(10, 5),
                                compression="gzip",
                                compression_opts=9)
    splitset = f.create_dataset("split",
                                (0, 1),
                                maxshape=(None, None),
                                dtype="int",
                                chunks=(10, 1),
                                compression="gzip",
                                compression_opts=9)
    return f, (srcset, zset, muset, logvarset, labelset, splitset)


def extract_from_dataset(model,
                         vocab,
                         cfg,
                         base_folder,
                         n_iter_num,
                         max_examples=20000):
    from data_processing.dataset import AttributeDataLoader
    model.eval()

    def extraction(iterator, max_examples, split_encoding):
        num_examples = min(max_examples, len(iterator.dataset))
        LOG.info("Extracting {} examples".format(num_examples))
        for batch in iter(iterator):
            with torch.no_grad():
                (mu, logvar), (z, c), dec_logits = model(
                    batch.text, q_c='classifier', sample_z='max')
            bsize = batch.text.shape[0]

            # Increase h5py set size
            extend_sets([srcset, zset, muset, logvarset, labelset, splitset], bsize)
            # Save to h5
            srcset[-bsize:] = batch.text
            zset[-bsize:] = z
            muset[-bsize:] = mu
            logvarset[-bsize:] = logvar
            ordered_labels = [
                getattr(batch, attr_name).numpy()
                for attr_name, _ in cfg.attributes]
            h5label = np.stack(ordered_labels, axis=1)
            labelset[-bsize:] = h5label
            h5split = np.array([split_encoding] * bsize).reshape(bsize, 1)
            splitset[-bsize:] = h5split
            if srcset.shape[0] >= max_examples:
                resize_sets([srcset, zset, muset, logvarset, labelset, splitset], max_examples)
                break

    # Set up dataset iterators
    LOG.info("Initializing dataloader")
    dataset = AttributeDataLoader(mbsize=cfg.vae.batch_size,
                                  max_seq_len=cfg.max_seq_len,
                                  device=torch.device('cpu'),
                                  attributes=cfg.attributes,
                                  **cfg.data_kwargs)
    assert dict(enumerate(dataset.TEXT.vocab.itos)) == vocab.ix2word, \
        'dataloader vocab  needs to match - specify with cfg.data_kwargs.fixed_vocab_path'
    # iteratorspecs = {s: {'subset': ['split='+s], 'repeat': False} for s in ['train', 'val', 'test']}
    iteratorspecs = {
        s: {
            'subset': ['split=' + s],
            'weighted_random_sample': True,
            'sample_prob_factors': cfg.amp_sample_prob_factors
        }
        for s in ['train', 'val', 'test']
    }
    iterators, _ = dataset.dataset.get_subset_iterators(iteratorspecs, cfg.vae.batch_size, torch.device('cpu'))
    split_encoding = {'train': 0, 'val': 1, 'test': 2}
    LOG.info("Set up storage and dataset. Extracting now...")
    for split, iterator in iterators.items():

        LOG.info("Encoding up to {} samples for split {}".format(max_examples, split))
        # Set up H5Py:
        # Set path and remove if file exists already
        path = "{}/states_{}_{}.h5".format(base_folder, split, n_iter_num)
        if os.path.isfile(path):
            os.remove(path)
        f, (srcset, zset, muset, logvarset, labelset, splitset) = \
            setup_hdf5_file(path, cfg)
        extraction(iterator, max_examples, split_encoding[split])
        f.close()


def build_faiss(h5_folder, n_iter_num):
    f_loc = os.path.join(h5_folder, "states_" + str(n_iter_num) + ".h5")
    f = h5py.File(f_loc, "r")
    LOG.info('Reading file at {} now...'.format(f_loc))
    data = f['z']

    num_seqs, z_dim = data.shape
    LOG.info("Processing {} sequences of size {}.".format(num_seqs, z_dim))

    # Initialize a new index
    index = faiss.IndexFlatIP(z_dim)

    # Fill it one batch at a time
    for ix in tqdm(range(0, num_seqs - 100, 100)):
        cdata = np.array(data[ix:ix + 100], dtype="float32")
        index.add(cdata)

    # Save and close
    f.close()
    faiss.write_index(index,
                      os.path.join(h5_folder,
                                   "index_" + str(n_iter_num) + ".faiss"))
