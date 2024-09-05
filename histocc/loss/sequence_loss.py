from torch import nn, Tensor


class Seq2SeqCrossEntropy(nn.Module):
    ''' Cross entropy loss for seq2seq models.

    Implemented as Module, i.e. the loss is the __call__ of an instance of the
    class.

    `yhat` is permuted [B, SEQ_LEN, VOCAB_SIZE] -> [SEQ_LEN, B, VOCAB_SIZE].
    Note that the first token (BoS) is not predicted, and so the SEQ_LEN is one
    shorter than the actual length of the sequence. For the same reason, the
    first element in each sequence in `target` is discarded.

    '''

    def __init__(self, pad_idx: int):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, yhat: Tensor, target: Tensor) -> Tensor: # pylint: disable=C0116
        return self.cross_entropy(yhat.permute(0, 2, 1), target[:, 1:])
