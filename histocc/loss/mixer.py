from torch import nn, Tensor


class LossMixer(nn.Module):
    def __init__(
            self,
            loss_fn_seq2seq: nn.Module,
            loss_fn_linear: nn.Module,
            seq2seq_weight: float = 0.5,
    ):
        super().__init__()

        self.loss_fn_seq2seq = loss_fn_seq2seq
        self.loss_fn_linear = loss_fn_linear

        self.seq2seq_weight = seq2seq_weight

    def forward(
            self,
            out_seq2seq: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 1, VOCAB]
            out_linear: Tensor, # [BATCH_SIZE, TOTAL_NUM_CODES]
            target_seq2seq: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 2]
            target_linear: Tensor, # [BATCH_SIZE, TOTAL_NUM_CODES]
    ) -> Tensor:
        loss_seq2seq = self.loss_fn_seq2seq(out_seq2seq, target_seq2seq)
        loss_linear = self.loss_fn_linear(out_linear, target_linear)

        loss = self.seq2seq_weight * loss_seq2seq + (1 - self.seq2seq_weight) * loss_linear

        return loss
