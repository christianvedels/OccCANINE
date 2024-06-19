'''
Formatter for seq2seq-based HISCO systems

'''


import numpy as np

from .constants import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX


MAP_HISCO_IDX = {
    str(hisco_char): hisco_char + SEP_IDX + 4 for hisco_char in range(-3, 10)
}
MAP_IDX_HISCO = {value: key for key, value in MAP_HISCO_IDX.items()}

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
        formatted_hisco: list[int],
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
            chunks.append('')
        else:
            chunks.append(clean_hisco(chunk, rev_mapping))

        start_idx = end_idx

    clean = sep_value.join(chunks)

    # TODO consider adding cycle consistency check

    return clean
