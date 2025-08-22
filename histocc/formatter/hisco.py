'''
Formatter for seq2seq-based HISCO systems

'''


import math
import warnings

from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from ..datasets import DATASETS

from .constants import PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX


MAP_HISCO_IDX = {
    str(hisco_char): hisco_char + SEP_IDX + 4 for hisco_char in range(-3, 10)
}
MAP_IDX_HISCO = {value: key for key, value in MAP_HISCO_IDX.items()}

# TODO we directly access below sets while passing above lookups as args
# Probably stick to same approach for both sets and lookups
_HISCO_SPECIAL_KEYS = {str(hisco_char) for hisco_char in range(-3, 0)}
_HISCO_SPECIAL_VALS = {MAP_HISCO_IDX[key] for key in _HISCO_SPECIAL_KEYS}


def format_hisco( # pylint: disable=C0116
        raw_hisco: str,
        mapping: dict[str, int],
        broadcast: bool = False,
        ) -> list[int]:
    if raw_hisco in _HISCO_SPECIAL_KEYS:
        # Occurs only for -1, -2, and -3 cases
        label = [mapping[raw_hisco]]

        if broadcast: # Broadcast 1 token to 5 tokens for fixed block sizes
            label *= 5

        return label

    # Unless in -1, -2, or -3 special case, code should be 5 chars
    assert len(raw_hisco) == 5, raw_hisco

    label = []

    for char in raw_hisco:
        label.append(mapping[char])

    return label


def clean_hisco( # pylint: disable=C0116
        formatted_hisco: list[int] | np.ndarray,
        rev_mapping: dict[int, str],
        ) -> str:
    if formatted_hisco[0] in _HISCO_SPECIAL_VALS:
        # If first token is one of special HISCO codes (-1, -2, -3),
        # disregard all following tokens
        return rev_mapping[formatted_hisco[0]]

    # Unless in -1, -2, or -3 special case, code should be 5 chars
    assert len(formatted_hisco) == 5, formatted_hisco

    cleaned = []

    for idx in formatted_hisco:
        cleaned.append(rev_mapping[idx])

    return ''.join(cleaned)


def format_hisco_seq( # pylint: disable=C0116
        raw_seq: str,
        max_num_codes: int,
        mapping: dict[str, int],
        sep_value: str = '',
) -> np.ndarray:
    if sep_value == '':
        seq = [raw_seq]
    else:
        seq = raw_seq.split(sep_value)

    label = [BOS_IDX]

    for hisco_code in seq:
        label.extend(format_hisco(hisco_code, mapping))
        label.append(SEP_IDX)

    # Right now there is an SEP_IDX after last word, replace with EOS_IDX
    label[-1] = EOS_IDX

    # Now pad to uniform length (+ 2 due to BOS and EOS)
    padding = 5 * max_num_codes + 2 - len(label)
    label.extend([PAD_IDX] * padding)

    label = np.array(label)

    # TODO consider adding a cycle consistency check here

    return label.astype('float')


def clean_hisco_seq( # pylint: disable=C0116
        raw_pred: np.ndarray,
        rev_mapping: dict[int, str],
        sep_value: str = '',
) -> str:
    # Strip all BOS tokens
    pred = raw_pred[raw_pred != BOS_IDX]

    # End at first EOS token
    if EOS_IDX in pred:
        first_eos = np.where(pred == EOS_IDX)[0][0]
    else:
        first_eos = -1

    pred = pred[:first_eos]

    # Loop over all sub-sequences as defined by SEP_IDX
    start_idx = 0

    chunks = []

    for idx_sep in np.where(pred == SEP_IDX)[0]:
        chunks.append(clean_hisco(pred[start_idx:idx_sep], rev_mapping))
        start_idx = idx_sep + 1

    chunks.append(clean_hisco(pred[start_idx:], rev_mapping))

    clean = sep_value.join(chunks)

    # TODO consider adding cycle consistency check

    return clean


def format_hisco_seq_blocky( # pylint: disable=C0116
        raw_seq: str,
        max_num_codes: int,
        mapping: dict[str, int],
        sep_value: str = '',
) -> np.ndarray:
    if sep_value == '':
        seq = [raw_seq]
    else:
        seq = raw_seq.split(sep_value)

    assert len(seq) <= max_num_codes, raw_seq

    label = [BOS_IDX]

    for hisco_code in seq:
        label.extend(format_hisco(hisco_code, mapping, broadcast=True))

    # Now pad till `max_num_codes` achieved
    padding = (max_num_codes - len(seq)) * 5
    label.extend([PAD_IDX] * padding)

    label.append(EOS_IDX)

    label = np.array(label)

    # TODO consider adding a cycle consistency check here

    return label.astype('float')


def clean_hisco_seq_blocky( # pylint: disable=C0116
        raw_pred: np.ndarray,
        num_blocks: int,
        rev_mapping: dict[int, str],
        sep_value: str = '',
) -> str:
    # Loop over all sub-sequences, here referred to as "chunks"
    start_idx = 1 # skip initial BOS

    chunks = []

    for _ in range(num_blocks):
        end_idx = start_idx + 5

        chunk = raw_pred[start_idx:end_idx]

        if (chunk == PAD_IDX).any():
            pass
        else:
            chunks.append(clean_hisco(chunk, rev_mapping))

        start_idx = end_idx

    clean = sep_value.join(chunks)

    # TODO consider adding cycle consistency check

    return clean


class BlockyHISCOFormatter: # TODO consider implementing base formatter class
    '''
    Formatter class to map HISCO codes into format suitable for seq2seq model.
    Always codes a HISCO code as 5 integers, even if the code is, e.g., '-1',
    as its purpose is to code it in a way where each HISCO code occupies same
    number of elements, hence the 'blocky' part of the class' name.

    Parameters
    ----------
    max_num_codes : int
        Maximum number of HISCO codes for input. To ensure fixed length of
        coded version, the output of `self.transform_label` always has length
        `max_num_codes * 5 + 2`, where `+ 2` arises due to BOS and EOS tokens
    map_char_idx : dict[str, int]
        Lookup to map from a character of a HISCO code to its corresponding
        integer when coded for a seq2seq model. This includes the 13 different
        characters (-3, -2, ..., 9) as well as certain special tokens such as
        BOS and EOS tokens.
    map_idx_char : dict[int, str]
        The reverse mapping of `map_char_idx`
    sep_value : str
        Character (or string, in principle) used to denote separation of multiple
        HISCO codes in input. If `'&'`, then an input `'12345&-1'` will be split
        into two parts, those being `'12345` and `'-1'`.

    '''
    code_len: int = 5

    # Pre-initialization declaration to show guaranteed attribute existence
    format_seq: Callable
    clean_seq: Callable
    lookup_hisco: dict[int, int]

    def __init__(
            self,
            max_num_codes: int,
            map_char_idx: dict[str, int],
            map_idx_char: dict[int, str],
            sep_value: str = '',
    ):
        self.max_num_codes = max_num_codes
        self.map_char_idx = map_char_idx
        self.map_idx_char = map_idx_char
        self.sep_value = sep_value

        # A sequence has maximum length of 5 (each HISCO code) times number of
        # HISCO codes + 2 due to BOS & EOS tokens
        self.block_size = 5
        self._max_seq_len = self.max_num_codes * self.block_size + 2

        self.initialize()

    def initialize(self) -> None: # pylint: disable=C0116
        self.format_seq = partial(
            format_hisco_seq_blocky,
            max_num_codes=self.max_num_codes,
            mapping=self.map_char_idx,
            sep_value=self.sep_value,
            )
        self.clean_seq = partial(
            clean_hisco_seq_blocky,
            num_blocks=self.max_num_codes,
            rev_mapping=self.map_idx_char,
            sep_value=self.sep_value,
            )

        # FIXME Dislike below pattern but approach with least impact
        # of existing code
        keys = DATASETS['keys']()
        self.lookup_hisco = dict(keys[['code', 'hisco']].values)

    def sanitize(self, raw_input: str | pd.DataFrame | pd.Series) -> str | None: # pylint: disable=C0116
        if isinstance(raw_input, str) or raw_input is None:
            return raw_input

        # FIXME Dislike below pattern but approach with least impact
        # of existing code
        sanitized = []

        for i in range(1, self.max_num_codes + 1):
            code = raw_input[f'code{i}']
            if hasattr(code, 'item'):
                code = code.item()
            if code is None or (isinstance(code, float) and math.isnan(code)):
                break
            hisco = str(self.lookup_hisco[int(float(code))])

            if len(hisco) == 4:
                # Mistakenly stripped leading zero due to int coding
                hisco = '0' + hisco

            sanitized.append(hisco)

        sanitized = self.sep_value.join(sanitized)

        return sanitized

    @property
    def max_seq_len(self) -> int: # pylint: disable=C0116
        # Max. seq. len.: {self.max_num_codes} * 5 + 2, since BOS and EOS token.
        return self._max_seq_len

    @property
    def num_classes(self) -> list[int]: # pylint: disable=C0116
        return [max(self.map_idx_char) + 1] * self._max_seq_len

    def transform_label(self, raw_input: str | pd.DataFrame | pd.Series) -> np.ndarray | None:
        '''
        Given a sequence of HISCO codes as defined by a str or a 1-row
        pd.DataFrame, return a representaion suitable for a seq2seq model.

        Parameters
        ----------
        raw_input : str | pd.DataFrame | pd.Series
            Either a string of HISCO codes, separated by `self.sep_value`, or a 1-row
            pd.DataFrame with columns `['code1', 'code2', ..., f'code{self.max_num_codes}]`,
            each an integer present as a key in `self.lookup_hisco`, or a pd.Series corresponding
            to such a 1-row pd.DataFrame

            If input is a string, its format, assuming `self.sep_value == '&'`, could be of
            the form '12345&34567&-1', '-3', '67890', i.e., consisting of either five digit
            codes or one of the special codes '-1', '-2', '-3'.

        Returns
        -------
        np.ndarray | None
            1D array of length `self.max_num_codes * 5 + 2` of floats, with each float
            representing a non-negative integer in the range 0-`self.map_idx_char)`.

        Examples
        -------
        When parameterized `self.max_num_codes = 5`, `map_char_idx = MAP_HISCO_IDX`,
        and `self.sep_value = '&'` and provided with an input `'12345&-1'`, returns
        an np.ndarray
            array([
              2.,  9., 10., 11., 12., 13.,  7.,  7.,  7.,  7.,  7.,  1.,  1.,
              1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
              3.])

        '''
        seq = self.sanitize(raw_input)

        if seq is None:
            return None

        return self.format_seq(seq)

    def clean_pred(self, raw_pred: np.ndarray) -> str:
        '''
        Reverses the transformation applied by `self.transform_label` to convert a sequence of
        encoded HISCO codes back into a readable string format.

        Note that this method will always return a string. It does not support transforming
        back to the 1-row pd.DataFrame format supported by `self.transform_label`.

        Parameters
        ----------
        raw_pred : np.ndarray
            A 1D numpy array of integers representing a sequence of encoded HISCO codes.
            This array is typically the output from a seq2seq model and has a length
            of `self.max_num_codes * 5 + 2`.

        Returns
        -------
        str
            A string of HISCO codes separated by `self.sep_value`, reconstructed from the
            encoded sequence.

        Examples
        --------
        When parameterized `self.max_num_codes = 5`, `map_idx_char = MAP_IDX_HISCO`,
        and `self.sep_value = '&'`, provided with an input array like
        `array([2.,  9., 10., 11., 12., 13.,  7.,  7.,  7.,  7.,  7.,  1.,  1.,  1.,  1.,
                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  3.])`, returns a string
        like `'12345&-1'`.

        '''
        return self.clean_seq(raw_pred)


# TODO consider implementing register decorator
def hisco_blocky5() -> BlockyHISCOFormatter: # pylint: disable=C0116
    formatter = BlockyHISCOFormatter(
        max_num_codes=5,
        map_char_idx=MAP_HISCO_IDX,
        map_idx_char=MAP_IDX_HISCO,
        sep_value='&',
    )

    return formatter


def blocky5() -> BlockyHISCOFormatter: # pylint: disable=C0116
    warnings.warn(DeprecationWarning('`blocky5` is being deprecated in favor of `hisco_blocky5`'))

    return hisco_blocky5()
