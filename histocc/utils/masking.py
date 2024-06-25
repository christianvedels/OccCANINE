"""
@author: sa-tsdj

"""


import torch

from torch import Tensor


def generate_square_subsequent_mask(
        seq_len: int,
        device: torch.device | str = 'cuda', # pylint: disable=E1101
        ) -> Tensor:
    """
    Create mask of target for seq2seq models. Mask of target is used to mask
    out future elements in sequence.

    Note that `target_input` is 1 element shorter than the sequences, as the last
    element of each sequence has been discarded, as the last element is not
    used as input. This also means `seq_len` is one less than the true length
    of the sequences.

    Parameters
    ----------
    seq_len : int
        Length of sequences - 1, as last element of each sequence is not used
        as input. This means `seq_len = SEQ_LEN - 1`.
    device : Union[torch.device, str], optional
        Which device to create tensors at. The default is 'cuda'.

    Returns
    -------
    Tensor
        Target mask of shape [SEQ_LEN - 1, SEQ_LEN - 1]. Example where
        SEQ_LEN - 1 == 3:
            [[0, -inf, -inf],
             [0,    0, -inf],
             [0,    0,    0]].

    """
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1) # pylint: disable=E1101
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def create_mask(
        target_input: Tensor,
        pad_idx: int,
        device: torch.device | str = 'cuda', # pylint: disable=E1101
        ) -> tuple[Tensor, Tensor]:
    """
    Create mask of target and padding for seq2seq models. Mask of target is
    used to mask out future elements in sequence and mask of padding is used to
    mask out elements placed at end of sequences to pad all sequences to same
    length.

    Note that `target_input` is 1 element shorter than the sequences, as the last
    element of each sequence has been discarded, as the last element is not
    used as input. This also means `seq_len` is one less than the true length
    of the sequences.

    Implementation based on https://pytorch.org/tutorials/beginner/translation_transformer.html

    Parameters
    ----------
    target_input : Tensor
        Target input of shape [BATCH_SIZE, SEQ_LEN - 1], the last element of
        each sequence discarded as that is not used as input.
    pad_idx : int
        Index used to denote padding.
    device : Union[torch.device, str], optional
        Which device to create tensors at. The default is 'cuda'.

    Returns
    -------
    target_mask : Tensor
        Target mask of shape [SEQ_LEN - 1, SEQ_LEN - 1]. Example where
        SEQ_LEN - 1 == 3:
            [[0, -inf, -inf],
             [0,    0, -inf],
             [0,    0,    0]]
    target_padding_mask : Tensor
        Padding mask of shape [BATCH_SIZE, SEQ_LEN - 1]. Example where
        BATCH_SIZE == 2, SEQ_LEN - 1 == 3, and the second sequence has padding
        on last element:
            [[FALSE, FALSE, FALSE],
             [FALSE, FALSE,  TRUE]]

    """
    seq_len = target_input.shape[1]

    target_mask = generate_square_subsequent_mask(seq_len, device)
    target_padding_mask = (target_input == pad_idx).to(dtype=target_mask.dtype)

    return target_mask, target_padding_mask
