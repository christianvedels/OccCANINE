import torch

from torch import nn, Tensor


class OrderInvariantSeq2SeqCrossEntropy(nn.Module):
    def __init__(
            self,
            pad_idx: int,
            nb_blocks: int = 5,
            block_size: int = 5,
            push_to_pad_label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.nb_blocks = nb_blocks
        self.block_size = block_size

        # Loss to push towards occupations, and where loss is
        # invariant towards the order of predicted "blocks"
        # TODO we can probably remove the ignore, as we should never hit
        # a case with any PAD elements in self._order_invariant_loss
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            reduction='none',
            )

        # Loss to push towards padding
        self._pad_push_weight: float = 0.05 # FIXME make arg, is hparam
        self.padding_cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=push_to_pad_label_smoothing,
            )
        padding_mask = torch.full(
            size=(1, self.nb_blocks * self.block_size),
            fill_value=self.pad_idx,
            )
        # We need to register the mask as a buffer to ensure it
        # is correctly moved to new device when loss module is
        # moved
        self.register_buffer('padding_mask', padding_mask)

    def _push_to_pad(
            self,
            yhat: Tensor, # [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
    ) -> Tensor:
        return self.padding_cross_entropy(
            yhat,
            self.padding_mask.repeat(yhat.size(0), 1),
        )

    def _order_invariant_loss(
            self,
            yhat: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS, VOCAB]
            target: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS]
    ) -> Tensor:
        losses = []

        for target_block in range(self.nb_blocks):
            # Look at target block and calculate loss
            # with respect to all candidate predictions

            start_idx = target_block * self.block_size
            end_idx = start_idx + self.block_size

            if (target[:, start_idx:end_idx] == self.pad_idx).all():
                # Only padding remains
                # TODO verify OK with break, i.e., there should be no
                # target HISCO code #3 if there is no #2, etc.
                break

            block_losses = []

            for candidate_block in range(self.nb_blocks):
                candidate_start_idx = candidate_block * self.block_size
                candidate_end_idx = candidate_start_idx + self.block_size

                block_losses.append(self.cross_entropy(
                    yhat[:, :, candidate_start_idx:candidate_end_idx],
                    target[:, start_idx:end_idx],
                ).mean(axis=1, keepdims=True))

            block_losses = torch.concat(block_losses, axis=1)
            block_losses, _ = block_losses.min(axis=1)
            block_loss = block_losses.mean()
            losses.append(block_loss)

        return sum(losses) / self.nb_blocks # scale to ensure order length invariane

    def forward(
            self,
            yhat: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 1, VOCAB]
            target: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 2]
    ) -> Tensor: # pylint: disable=C0116
        # We discard the initial BOS token and the final
        # EOS token
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
        target = target[:, 1:-1] # ignore initial BOS and final EOS

        order_invariant_loss = self._order_invariant_loss(yhat, target)
        push_to_pad_loss = self._push_to_pad(yhat)

        loss = order_invariant_loss + self._pad_push_weight * push_to_pad_loss

        return loss
