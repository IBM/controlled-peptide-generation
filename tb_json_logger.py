import os
import json
import warnings

#orig: __all__ = ['Logger', 'configure', 'log_value', 'log_histogram', 'log_images']
from tensorboard_logger.tensorboard_logger import Logger, configure, log_value, log_histogram, log_images

_default_logger = None  # type: Logger
_log_dic = {} # {..,  it: {metric_name: val, metric_name: val, ...}, ...}


def configure(logdir, json_fn=None, flush_secs=2):
    """ Configure logging: a file will be written to logdir, and flushed
    every flush_secs.
    """
    global _default_logger, _log_dic
    if _default_logger is not None:
        raise ValueError('default logger already configured')
    _default_logger = Logger(logdir, flush_secs=flush_secs)
    if _log_dic:
        raise ValueError('_log_dic not empty! ' + str(_log_dic))
    if json_fn and os.path.exists(json_fn):
        try:
            with open(json_fn) as fh:
                _log_dic.update({e['it']: e for e in json.load(fh)})
        except json.decoder.JSONDecodeError as e:
            warnings.warn('Couldnt decode {}: {}'.format(json_fn, str(e)))


def _check_default_logger():
    if _default_logger is None:
        raise ValueError(
            'default logger is not configured. '
            'Call tensorboard_logger.configure(logdir), '
            'or use tensorboard_logger.Logger')


def log_value(name, value, step=None):
    global _default_logger, _log_dic
    _check_default_logger()
    _default_logger.log_value(name, value, step=step)
    assert not _log_dic or step >= max(_log_dic.keys()), \
        'logging into the past: {} < {}'.format(step, max(_log_dic.keys()))
    _log_dic.setdefault(step, {'it': step})
    _log_dic[step][name] = float(value)


# histogram and images: ignore for json
def log_histogram(name, value, step=None):
    global _default_logger
    _check_default_logger()
    _default_logger.log_histogram(name, value, step=step)


def log_images(name, images, step=None):
    global _default_logger
    _check_default_logger()
    _default_logger.log_images(name, images, step=step)


def get_logged_values(step):
    return _log_dic[step]


def get_last_logged_values():
    if not _log_dic:
        return {}
    step = max(_log_dic.keys())
    return get_logged_values(step)


def export_to_json(json_fn, it_filter=lambda k,v: True, trunc_tail=None, write_empty=False):
    # make it into a ordered list matching HPO expectation
    global _log_dic
    if trunc_tail and _log_dic:
        last_it = max(_log_dic.keys())
        tail_filter = lambda it: it >= last_it - trunc_tail
    else:
        tail_filter = lambda it: True
    for_export = [_log_dic[it] for it in sorted(_log_dic.keys()) if it_filter(it, _log_dic[it]) and tail_filter(it)]

    if for_export or write_empty:
        with open(json_fn, 'w') as fh:
            json.dump(for_export, fh, indent=1)


log_value.__doc__ = Logger.log_value.__doc__