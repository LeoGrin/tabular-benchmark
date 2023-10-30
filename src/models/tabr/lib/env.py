# NOTE: this file must not import anything from lib.

import os
from pathlib import Path
from typing import Union

PROJECT_DIR = Path(os.environ['PROJECT_DIR']).absolute().resolve()
CACHE_DIR = PROJECT_DIR / 'cache'
DATA_DIR = PROJECT_DIR / 'data'
EXP_DIR = PROJECT_DIR / 'exp'

assert PROJECT_DIR.exists()
CACHE_DIR.mkdir(exist_ok=True)


def get_path(path: Union[str, Path]) -> Path:
    path = str(path)
    if path.startswith(':'):
        path = PROJECT_DIR / path[1:]
    return Path(path).absolute().resolve()


def try_get_relative_path(path: Union[str, Path]) -> Path:
    path = get_path(path)
    return path.relative_to(PROJECT_DIR) if PROJECT_DIR in path.parents else path
