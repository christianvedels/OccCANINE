import torch

from torch import Tensor


class Averager:
    def __init__(self):
        self.sum: int | float = 0
        self.count: int = 0

    @property
    def avg(self) -> float:
        return self.sum / self.count

    def update(self, val: int | float, num: int = 1):
        self.sum += val * num
        self.count += num


def seq2seq_sequence_accuracy(
        output: Tensor,
        target: Tensor,
        pad_idx: int,
        ) -> tuple[Tensor, Tensor]:
    """
    Calculate the sequence and token accuracies of a seq2seq-style output and
    target. In both calculations, all indexes of `target` that correspond to
    padding are omitted from the calculations. Note that the token accuracy
    weighs longer sequences higher (proportional to length). However, as this
    function is likely called on batches (rather than entire data set), this
    will only be true in a batch-wise sense and the final token accuracy will
    weigh observations proportional to their length within batches but
    uniformly across batches.

    Parameters
    ----------
    output : Tensor
        [B, SeqLen, Features]-shaped model output.
    target : Tensor
        [B, SeqLen]-shaped targets.
    pad_idx : int
        Which index used to represent padding.

    Returns
    -------
    (Tensor, Tensor)
        1-element tensors with the sequence and token accuracy, respectively.

    """
    max_index = torch.argmax(output, dim=2).detach().cpu() # pylint: disable=E1101
    _target = target.clone() # To allow in-place operations to not change input

    # Padded elements should not be accounted for in calculation; set to -1 to
    # ensure they match, allowing the calculation of sequence accuracy used
    target_padding_mask = (_target == pad_idx)
    max_index[target_padding_mask] = -1
    _target[target_padding_mask] = -1

    is_eq = max_index == _target.detach().cpu()
    seq_acc = 100 * is_eq.all(axis=1).float().mean()

    # When calculating token accuracy, make sure not to inflate with the padded
    # values; hence remove those from both numerator and enumerator
    token_acc = 100 * (is_eq.float().sum() - target_padding_mask.sum()) / (~target_padding_mask).sum()

    return seq_acc, token_acc


def _get_best_block_hits(
        max_index: Tensor, # [BATCH_SIZE, SEQ-LEN]
        target_block: Tensor, # [BATCH_SIZE, BLOCK-LEN]
        nb_blocks: int,
        block_size: int,
) -> Tensor:
    hits = []

    for candidate_block in range(nb_blocks):
        candidate_start_idx = candidate_block * block_size
        candidate_end_idx = candidate_start_idx + block_size

        mask = max_index[:, candidate_start_idx:candidate_end_idx] == target_block
        score = mask.double().mean(axis=1, keepdims=True)

        hits.append(score)

    hits = torch.concat(hits, axis=1)
    hits, _ = hits.max(axis=1, keepdims=True)

    return hits


def order_invariant_accuracy(
        output: Tensor,
        target: Tensor,
        pad_idx: int,
        nb_blocks: int = 5,
        block_size: int = 5,
) -> tuple[Tensor, Tensor]:
    max_index = torch.argmax(output, dim=2).detach().cpu()
    _target = target.detach().cpu()

    # FIXME currently, we treat a prediction which cover all labels
    # but ALSO predict a code which is NOT in the set of labels as
    # correct. We probably want to treat as a case an incorrect to
    # not inflate accuracy

    # TODO consider implementation which discard PAD_IDX
    # values token accuracy to get more meaningful metric

    hits = []

    for target_block in range(nb_blocks):
        start_idx = target_block * block_size
        end_idx = start_idx + block_size

        block_hits = _get_best_block_hits(
            max_index=max_index,
            target_block=_target[:, start_idx:end_idx],
            nb_blocks=nb_blocks,
            block_size=block_size,
        )
        hits.append(block_hits)

    hits = torch.concat(hits, axis=1)

    # Each block has same size, hence we can directly average over all
    # rows and blocks at once
    token_acc = 100 * hits.mean()

    # Sequence classified as correct if each target block have a 100%
    # match with some candidate block
    seq_acc = 100 * (hits == 1.0).all(axis=1).double().mean()

    # TODO need to account for `pad_idx`, at the moment treated equally
    # with any "real" token

    return seq_acc, token_acc
