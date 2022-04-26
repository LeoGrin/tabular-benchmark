import json
import sys

import numpy as np
import wget


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(
                obj,
                (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                 np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert_unserialized_data_to_serialized_json(data):
    return json.dumps(data, cls=NumpyEncoder)


def stack_ragged(array_list, axis=1):
    """
    Stack ragged data table for downstream npz storage.
    From https://tonysyu.github.io/ragged-arrays.html
    """
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx


def bar_progress(current, total, width=80):
    """Display progress bar while downloading.

    https://stackoverflow.com/questions/58125279/
    python-wget-module-doesnt-show-progress-bar"
    """

    progress_message = (
        f'Downloading: {current/total * 100:.0f} %% '
        f'[{current:.2e} / {total:.2e}] bytes')

    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download(paths, urls):
    # Download URLs to paths.

    if not isinstance(urls, list):
        urls = [urls]

    if not isinstance(paths, list):
        paths = [paths]

    if not len(urls) == len(paths):
        raise ValueError('Need exactly one path per URL.')

    for path, url in zip(paths, urls):
        print(f'Downloading {url}.')
        path.parent.mkdir(parents=True, exist_ok=True)
        wget.download(url, out=str(path), bar=bar_progress)
