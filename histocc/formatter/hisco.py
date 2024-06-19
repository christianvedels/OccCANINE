'''
Formatter for seq2seq-based HISCO systems

'''


import math

from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from histocc import DATASETS

from .constants import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX


MAP_HISCO_IDX = {
    str(hisco_char): hisco_char + SEP_IDX + 4 for hisco_char in range(-3, 10)
}
MAP_IDX_HISCO = {value: key for key, value in MAP_HISCO_IDX.items()}

# TODO we directly access below sets while passing above lookups as args
# Probably stick to same approach for both sets and lookups
_HISCO_SPECIAL_KEYS = {str(hisco_char) for hisco_char in range(-3, 0)}
_HISCO_SPECIAL_VALS = {MAP_HISCO_IDX[key] for key in _HISCO_SPECIAL_KEYS}


def format_hisco(
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


def clean_hisco(
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


def format_hisco_seq(
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


def clean_hisco_seq(
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


def format_hisco_seq_blocky(
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


def clean_hisco_seq_blocky(
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

        if chunk[0] == PAD_IDX:
            pass
        else:
            chunks.append(clean_hisco(chunk, rev_mapping))

        start_idx = end_idx

    clean = sep_value.join(chunks)

    # TODO consider adding cycle consistency check

    return clean


class BlockyHISCOFormatter: # TODO consider implementing base formatter class
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

        # A sequence has maximum length of 5 (each HISCO code) times number of HISCO codes + 2 due to BOS & EOS tokens
        self._max_seq_len = self.max_num_codes * 5 + 2

        self.initialize()

    def initialize(self) -> None:
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

    def sanitize(self, raw_input: str | pd.DataFrame) -> str | None:
        if isinstance(raw_input, str) or raw_input is None:
            return raw_input

        # FIXME Dislike below pattern but approach with least impact
        # of existing code
        sanitized = []

        for i in range(1, self.max_num_codes + 1):
            code = raw_input[f'code{i}'].item()

            if code is None or math.isnan(code):
                # If hit NaN, assume subsequent values are also NaN
                break

            hisco = self.lookup_hisco[code]
            hisco = str(hisco)

            if len(hisco) == 4:
                # Mistakenly stripped leading zero due to int coding
                hisco = '0' + hisco

            sanitized.append(hisco)

        sanitized = self.sep_value.join(sanitized)

        return sanitized

    @property
    def max_seq_len(self) -> int:
        print(f'Max. seq. len.: {self.max_num_codes} * 5 + 2, since BOS and EOS token.')
        return self._max_seq_len

    @property
    def num_classes(self) -> list[int]:
        return [max(self.map_idx_char) + 1] * self._max_seq_len

    def transform_label(self, raw_input: str | pd.DataFrame) -> np.ndarray | None:
        seq = self.sanitize(raw_input)

        if seq is None:
            return None

        return self.format_seq(seq)

    def clean_pred(self, raw_pred: np.ndarray) -> str:
        clean = self.clean_seq(raw_pred)

        return clean


# TODO consider implementing register decorator
def blocky5() -> BlockyHISCOFormatter:
    formatter = BlockyHISCOFormatter(
        max_num_codes=5,
        map_char_idx=MAP_HISCO_IDX,
        map_idx_char=MAP_IDX_HISCO,
        sep_value='&',
    )

    return formatter
