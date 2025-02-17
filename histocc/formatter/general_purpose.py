'''
General-purpose formatter that attempts to be adaptable to a wide range of
settings. This is particularly useful for fine-tuning, where we want a for-
matter that people can use right out of the box, even if they use a new sys-
tem or have some errors in their label data.

'''


import math
import string

from functools import partial

from typing import Callable

import numpy as np
import pandas as pd

from .constants import PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX


def build_mapping(chars: list[int | str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = [str(x) for x in chars]

    if not len(chars) == len(set(chars)):
        raise ValueError(f'Duplicate values in character-set: {chars}')

    for char in chars:
        if len(char) > 1:
            raise ValueError(f'Multi-character value in character-set: {chars}')

    map_char_idx = {
        str(x): i + SEP_IDX + 1 for i, x in enumerate(chars)
    }
    map_idx_char = {val: key for key, val in map_char_idx.items()}

    return map_char_idx, map_idx_char


def build_multichar_mapping(chars: list[int | str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = [str(x) for x in chars]

    if not len(chars) == len(set(chars)):
        raise ValueError(f'Duplicate values in character-set: {chars}')

    map_char_idx = {
        str(x): i + SEP_IDX + 1 for i, x in enumerate(chars)
    }
    map_idx_char = {val: key for key, val in map_char_idx.items()}

    return map_char_idx, map_idx_char


def format_code(
        raw_code: str,
        mapping: dict[str, int],
        block_size: int,
        within_block_sep: str | None = None,
        ) -> list[int]:
    label = []

    if within_block_sep is not None:
        raw_code = raw_code.split(within_block_sep)

    for char in raw_code:
        label.append(mapping[char])

    if len(label) > block_size:
        raise ValueError(f'Code chunk {raw_code} longer than block size ({block_size})')

    padding = block_size - len(label)
    label.extend([EOS_IDX] * padding)

    return label


def clean_code(
        formatted_code: list[int] | np.ndarray,
        rev_mapping: dict[int, str],
        within_block_sep: str | None = None,
        ) -> str:
    cleaned = []

    for idx in formatted_code:
        if idx in {PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX}:
            continue

        cleaned.append(rev_mapping[idx])

    if within_block_sep is not None:
        join_char = within_block_sep
    else:
        join_char = ''

    return join_char.join(cleaned)


def format_code_seq_blocky(
        raw_seq: str,
        max_num_codes: int,
        block_size: int,
        mapping: dict[str, int],
        sep_value: str = '',
        within_block_sep: str | None = None,
) -> np.ndarray:
    if sep_value == '':
        seq = [raw_seq]
    else:
        seq = raw_seq.split(sep_value)

    assert len(seq) <= max_num_codes, raw_seq

    label = [BOS_IDX]

    for code in seq:
        label.extend(format_code(code, mapping, block_size, within_block_sep))

    # Now pad till `max_num_codes` achieved
    padding = (max_num_codes - len(seq)) * block_size
    label.extend([PAD_IDX] * padding)

    label.append(EOS_IDX)

    label = np.array(label)

    return label.astype('float')


def clean_code_seq_blocky( # pylint: disable=C0116
        raw_pred: np.ndarray,
        num_blocks: int,
        block_size: int,
        rev_mapping: dict[int, str],
        sep_value: str = '',
        within_block_sep: str | None = None,
) -> str:
    # Loop over all sub-sequences, here referred to as "chunks"
    start_idx = 1 # skip initial BOS

    chunks = []

    for _ in range(num_blocks):
        end_idx = start_idx + block_size

        chunk = raw_pred[start_idx:end_idx]

        if (chunk == PAD_IDX).all():
            pass
        else:
            chunks.append(clean_code(chunk, rev_mapping, within_block_sep))

        start_idx = end_idx

    clean = sep_value.join(chunks)

    return clean


class BlockyFormatter:
    '''
    General-purpose formatter class to map a wide array of occupational code
    formats into a format suiteable for seq2seq models. Codes each code as
    `block_size` integers, as its purpose is to code it in a way where each
    code occupies same number of elements, hence the 'blocky' part of the class'
    name.

    Parameters
    ----------
    target_cols : list[str]
        List of column names containing labels. May contain multiple strings
        since multiple occupational codes may be present. The length of this
        list determines the maximum number of codes, and even if fewer codes
        are present, padding is used to ensure a consisten encoding length.
    block_size : int
        The size of each code, i.e., the number of characters is maximally
        takes up. Each code shorter is padded to ensure uniform length.
    map_char_idx : dict[str, int]
        Lookup to map from a character of a code to its corresponding integer
        when coded for a seq2seq model.
    map_idx_char : dict[int, str]
        The reverse mapping of `map_char_idx`
    sep_value : str
        Character (or string, in principle) used to denote separation of multiple
        codes in input. If `'&'`, then an input `'123&456'` will be split into two
        parts, those being `'123` and `'456'`.
    within_block_sep : str | None
        Optional string to signify separation of tokens *within* each block. This
        is useful to allow multi-character tokens, such as "-1" (HISCO) or "14"
        (PTSI). If not specified, each character is treated as its own token.

    '''
    # Pre-initialization declaration to show guaranteed attribute existence
    format_seq: Callable
    clean_seq: Callable

    def __init__(
            self,
            target_cols: list[str],
            block_size: int,
            map_char_idx: dict[str, int],
            map_idx_char: dict[int, str],
            sep_value: str = '',
            within_block_sep: str | None = None,
    ):
        self.target_cols = target_cols
        self.max_num_codes = len(target_cols)
        self.block_size = block_size

        self.map_char_idx = map_char_idx
        self.map_idx_char = map_idx_char

        # TODO verify certain things are not in mapping, e.g.:
        # ' '
        # `sep_value`
        # maybe `within_block_sep`, but not sure

        if within_block_sep is not None and within_block_sep == sep_value:
            raise ValueError('Cannot use same separator for between as within chunks ({sep_value} == {within_block_sep})')

        self.sep_value = sep_value
        self.within_block_sep = within_block_sep

        # Codes have length self.block_size and we add BOS and EOS tokens
        self._max_seq_len = self.max_num_codes * self.block_size + 2

        self.initialize()

    def __repr__(self):
        return (
            f"BlockyFormatter(\n"
            f"  target_cols={self.target_cols},\n"
            f"  max_num_codes={self.max_num_codes},\n"
            f"  block_size={self.block_size},\n"
            f"  sep_value='{self.sep_value}',\n"
            f"  within_block_sep={self.within_block_sep},\n"
            f"  max_seq_len={self._max_seq_len},\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )

    def initialize(self) -> None: # pylint: disable=C0116
        self.format_seq = partial(
            format_code_seq_blocky,
            max_num_codes=self.max_num_codes,
            block_size=self.block_size,
            mapping=self.map_char_idx,
            sep_value=self.sep_value,
            within_block_sep=self.within_block_sep,
            )
        self.clean_seq = partial(
            clean_code_seq_blocky,
            num_blocks=self.max_num_codes,
            block_size=self.block_size,
            rev_mapping=self.map_idx_char,
            sep_value=self.sep_value,
            within_block_sep=self.within_block_sep,
            )

    def sanitize(self, raw_input: str | pd.DataFrame | pd.Series) -> str | None: # pylint: disable=C0116
        if isinstance(raw_input, str) or raw_input is None:
            return raw_input

        sanitized = []

        for target_col in self.target_cols:
            code = raw_input[target_col]

            if isinstance(code, pd.Series):
                code = code.item()

            if code is None:
                # If hit None, assume subsequent values are also None
                break

            if isinstance(code, float):
                # Due to missings/None/NaN, type might be float. In such
                # cases, we want to either break if NaN or convert back
                # to integer
                if math.isnan(code):
                    break

                code = int(code)

            code = str(code)

            sanitized.append(code)

        sanitized = self.sep_value.join(sanitized)

        return sanitized

    @property
    def max_seq_len(self) -> int: # pylint: disable=C0116
        return self._max_seq_len

    @property
    def num_classes(self) -> list[int]: # pylint: disable=C0116
        return [max(self.map_idx_char) + 1] * self._max_seq_len

    def transform_label(self, raw_input: str | pd.DataFrame | pd.Series) -> np.ndarray | None:
        '''
        Given a sequence of codes as defined by a str or a 1-row
        pd.DataFrame, return a representaion suitable for a seq2seq model.

        Parameters
        ----------
        raw_input : str | pd.DataFrame | pd.Series
            Either a string of codes, separated by `self.sep_value`, or a 1-row
            pd.DataFrame (or pd.Series) with columns `self.target_cols`,
            each an (a) integer representing a code or (b) string representing a code OR
            being a space (' '), which is interpeted as "no code" OR None which is inter-
            preted as "no code".
            If input is a string, its format, assuming `self.sep_value == '&'`, could be of
            the form '123&456&', '123', '567', i.e., consisting of one or multiple three
            digit codes (depending on `self.map_char_idx`, `self.block_size`, ...).

        Returns
        -------
        np.ndarray | None
            1D array of length `self.max_num_codes * self.block_size + 2` of floats, with each float
            representing a non-negative integer in the range 0-`max(self.map_idx_char) + 1`.

        Examples
        -------
        When parameterized `self.max_num_codes = 2`, `map_char_idx = occ1950.MAP_OCC1950_IDX`,
        `block_size = 2`, and `self.sep_value = '&'` and provided with an input `'123&456'`,
        returns an np.ndarray
            array([
              float(BOS_IDX),  6.,  7.,  8.,  9., 10., 11., float(EOS_IDX),
              ])

        '''
        seq = self.sanitize(raw_input)

        if seq is None:
            return None

        return self.format_seq(seq)

    def clean_pred(self, raw_pred: np.ndarray) -> str:
        return self.clean_seq(raw_pred)


def construct_finetune_formatter(
        block_size: int,
        target_cols: list[str],
        chars: list[int | str] | None = None,
) -> BlockyFormatter:
    if chars is None:
        chars = list(range(0, 10)) + ['-']

    map_char_idx, map_idx_char = build_mapping(chars)

    formatter = BlockyFormatter(
        target_cols=target_cols,
        block_size=block_size,
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        sep_value='&',
    )

    return formatter


def construct_general_purpose_formatter(
        block_size: int,
        target_cols: list[str],
        chars: list[int | str] | None = None,
        use_within_block_sep: bool = False,
) -> BlockyFormatter:
    _sep_token = '&'
    _within_block_sep_token = ','

    if use_within_block_sep:
        within_block_sep = _within_block_sep_token
    else:
        within_block_sep = None

    if chars is None:
        chars = list(range(-999, 1000))
        chars += list(string.ascii_letters)
        chars += list(string.punctuation)

        chars = [c for c in chars if not c in (_sep_token, _within_block_sep_token)]

    map_char_idx, map_idx_char = build_multichar_mapping(chars)

    formatter = BlockyFormatter(
        target_cols=target_cols,
        block_size=block_size,
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        sep_value=_sep_token,
        within_block_sep=within_block_sep,
    )

    return formatter
