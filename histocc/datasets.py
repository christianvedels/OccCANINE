'''
Loader fns for embedded datasets.

'''

from importlib.resources import files

import pandas as pd


__ALL__ = ['DATASETS']

def _load_keys() -> pd.DataFrame:
    fn_keys = files('histocc').joinpath('Data/Key.csv')

    with fn_keys.open() as file:
        keys = pd.read_csv(file, skiprows=[1])

    return keys


def _load_toydata() -> pd.DataFrame:
    fn_keys = files('histocc').joinpath('Data/TOYDATA.csv')

    with fn_keys.open() as file:
        keys = pd.read_csv(file)

    return keys


DATASETS = {
    'keys': _load_keys,
    'toydata': _load_toydata,
}
