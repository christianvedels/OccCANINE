'''
Formatter for seq2seq-based IPUMS OCC1950 systems.

See https://usa.ipums.org/usa-action/variables/OCC1950#codes_section

'''


from functools import partial

from typing import Callable

import numpy as np
import pandas as pd

from .constants import PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX


CODE_LEN: int = 3
MAP_OCC1950_IDX = {
    str(occ1950_char): occ1950_char + SEP_IDX + 1 for occ1950_char in range(10)
}
MAP_IDX_OCC1950 = {value: key for key, value in MAP_OCC1950_IDX.items()}


def format_occ1950(
        raw_occ1950: str,
        mapping: dict[str, int],
        ) -> list[int]:
    assert len(raw_occ1950) == CODE_LEN, raw_occ1950

    label = []

    for char in raw_occ1950:
        label.append(mapping[char])

    return label


def clean_occ1950(
        formatted_occ1950: list[int] | np.ndarray,
        rev_mapping: dict[int, str],
        ) -> str:
    assert len(formatted_occ1950) == CODE_LEN, formatted_occ1950

    cleaned = []

    for idx in formatted_occ1950:
        cleaned.append(rev_mapping[idx])

    return ''.join(cleaned)


def format_occ1950_seq_blocky(
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

    for occ1950_code in seq:
        label.extend(format_occ1950(occ1950_code, mapping))

    # Now pad till `max_num_codes` achieved
    padding = (max_num_codes - len(seq)) * CODE_LEN
    label.extend([PAD_IDX] * padding)

    label.append(EOS_IDX)

    label = np.array(label)

    return label.astype('float')


def clean_occ1950_seq_blocky( # pylint: disable=C0116
        raw_pred: np.ndarray,
        num_blocks: int,
        rev_mapping: dict[int, str],
        sep_value: str = '',
) -> str:
    # Loop over all sub-sequences, here referred to as "chunks"
    start_idx = 1 # skip initial BOS

    chunks = []

    for _ in range(num_blocks):
        end_idx = start_idx + CODE_LEN

        chunk = raw_pred[start_idx:end_idx]

        if (chunk == PAD_IDX).any():
            pass
        else:
            chunks.append(clean_occ1950(chunk, rev_mapping))

        start_idx = end_idx

    clean = sep_value.join(chunks)

    return clean


class BlockyOCC1950Formatter:
    '''
    Formatter class to map OCC1950 codes into format suitable for seq2seq model.
    Always codes an OCC1950 code as 3 integers, as its purpose is to code it in a
    way where each OCC1950 code occupies same number of elements, hence the 'blocky'
    part of the class' name.

    Parameters
    ----------
    max_num_codes : int
        Maximum number of OCC1950 codes for input. To ensure fixed length of
        coded version, the output of `self.transform_label` always has length
        `max_num_codes * CODE_LEN + 2`, where `+ 2` arises due to BOS and EOS
        tokens.
    map_char_idx : dict[str, int]
        Lookup to map from a character of a OCC1950 code to its corresponding
        integer when coded for a seq2seq model. This includes the 10 different
        characters (0, 1, ..., 9) as well as certain special tokens such as
        BOS and EOS tokens.
    map_idx_char : dict[int, str]
        The reverse mapping of `map_char_idx`
    sep_value : str
        Character (or string, in principle) used to denote separation of multiple
        OCC1950 codes in input. If `'&'`, then an input `'123&456'` will be split
        into two parts, those being `'123` and `'456'`.

    '''
    code_len: int = CODE_LEN

    # Pre-initialization declaration to show guaranteed attribute existence
    format_seq: Callable
    clean_seq: Callable

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

        # Code have length CODE_LEN and we add BOS and EOS tokens
        self._max_seq_len = self.max_num_codes * CODE_LEN + 2

        self.initialize()

    def initialize(self) -> None: # pylint: disable=C0116
        self.format_seq = partial(
            format_occ1950_seq_blocky,
            max_num_codes=self.max_num_codes,
            mapping=self.map_char_idx,
            sep_value=self.sep_value,
            )
        self.clean_seq = partial(
            clean_occ1950_seq_blocky,
            num_blocks=self.max_num_codes,
            rev_mapping=self.map_idx_char,
            sep_value=self.sep_value,
            )

    def sanitize(self, raw_input: str | pd.DataFrame | pd.Series) -> str | None: # pylint: disable=C0116
        if isinstance(raw_input, str) or raw_input is None:
            return raw_input

        sanitized = []

        for i in range(1, self.max_num_codes + 1):
            occ1950 = raw_input[f'OCC1950_{i}']

            if isinstance(occ1950, pd.Series):
                occ1950 = occ1950.item()

            # NOTE: Codes which should be missing may be wrongly coded
            # as a single space
            # This additionally means that the entire column with such
            # cases is then coded as strings, incosistent with columns
            # that may be coded as integers
            if isinstance(occ1950, str) and occ1950 == ' ':
                occ1950 = None

            if occ1950 is None:
                # If hit None, assume subsequent values are also None
                break

            occ1950 = str(occ1950)

            if len(occ1950) < CODE_LEN:
                # Mistakenly stripped leading zero(s) due to int coding
                occ1950 = '0' * (CODE_LEN - len(occ1950)) + occ1950

            sanitized.append(occ1950)

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
        Given a sequence of OCC1950 codes as defined by a str or a 1-row
        pd.DataFrame, return a representaion suitable for a seq2seq model.

        Parameters
        ----------
        raw_input : str | pd.DataFrame | pd.Series
            Either a string of OCC1950 codes, separated by `self.sep_value`, or a 1-row
            pd.DataFrame (or pd.Series) with columns `['OCC1950_1', 'OCC1950_2', ...]`,
            each an (a) integer representing a code or (b) string representing a code OR
            being a space (' '), which is interpeted as "no code" OR None which is inter-
            preted as "no code".
            If input is a string, its format, assuming `self.sep_value == '&'`, could be of
            the form '123&456&', '123', '567', i.e., consisting of one or multiple three
            digit codes.

        Returns
        -------
        np.ndarray | None
            1D array of length `self.max_num_codes * CODE_LEN + 2` of floats, with each float
            representing a non-negative integer in the range 0-`max(self.map_idx_char) + 1`.

        Examples
        -------
        When parameterized `self.max_num_codes = 2`, `map_char_idx = MAP_OCC1950_IDX`,
        and `self.sep_value = '&'` and provided with an input `'123&456'`, returns
        an np.ndarray
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


def occ1950_blocky1() -> BlockyOCC1950Formatter: # pylint: disable=C0116
    formatter = BlockyOCC1950Formatter(
        max_num_codes=1,
        map_char_idx=MAP_OCC1950_IDX,
        map_idx_char=MAP_IDX_OCC1950,
        sep_value='&',
    )

    return formatter


def occ1950_blocky2() -> BlockyOCC1950Formatter: # pylint: disable=C0116
    formatter = BlockyOCC1950Formatter(
        max_num_codes=2,
        map_char_idx=MAP_OCC1950_IDX,
        map_idx_char=MAP_IDX_OCC1950,
        sep_value='&',
    )

    return formatter
