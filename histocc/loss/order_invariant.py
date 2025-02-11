'''
Implementation of loss function for seq2seq model which is
invariant to the order of which "blocks" (i.e., HISCO codes)
are predicted.

The motivation is that we do not particularly care whether
a model predicts whether an occupational description corresponds
to
    1) Farmer & fisher
    2) Fisher & farmer
Further, we cannot guarantee that our labels adhere to some well-
defined structure which makes the order of labels inferable from
the occupational description.

'''

import torch
from torch import nn, Tensor

class OrderInvariantSeq2SeqCrossEntropy(nn.Module):
    '''
    seq2seq-style loss function for sequences consisting of
    `nb_blocks` blocks, all of size `block_size`. Note that
    combining this with teacher forcing may lead to undesirable
    behavior where the model learns to condition on the label
    when predicting it, by placing its prediction in a later
    block than the associated label, see discussion in
    https://github.com/christianvedels/OccCANINE/issues/79

    The module consists of two types of losses:

    1) BLOCK ORDER-INVARIANT CLASSIFICATION
    For each target block, calculate the loss with respect
    to all `nb_blocks` candidate blocks, using ordinary
    cross entropy as the loss function. Then choose the
    smallest of the `nb_blocks` candidate losses as the loss
    with respect to the given target block. Do so for each
    of the `nb_blocks` target blocks and average these to
    obtain the final loss related to the block order-invariant
    classification loss.

    Note that the above procedure is calculated separately
    for each observation in the input batch, i.e., for the
    first target block, two different observations in the same
    batch may end up selecting different candidate blocks.

    Additionally, since most targets consist mainly of empty
    blocks, which are denoted using the `pad_idx` value,
    we ignore its index when calculating the loss. This ensures
    that the loss for the second target block only depends on
    the observations in the batch for which the second target
    block is not empty.

    2) PUSH TOWARDS SPARSITY
    To ensure the model is pushed towards only producing as
    many candidates as there are actual non-empty target blocks,
    apply cross entropy with respect to the entire sequence,
    using `pad_idx` as the target value.

    Parameters
    ----------
    pad_idx : int
        Which index used to represent padding.
    nb_blocks : int
        Number of blocks. Excluding BOS and EOS tokens, each
        sequence has length `nb_blocks * block_size`.
    block_size : int
        Number of elements in each block. Excluding BOS and EOS
        tokens, each sequence has length `nb_blocks * block_size`.
    push_to_pad_scale_factor : float
        Scaling factor applied to the second part of the loss.
        The total loss consists of the first part of the loss
        added with this times the second part of the loss.
    push_to_pad_label_smoothing : float
        Label smoothing factor applied to the second part of the
        loss.

    '''
    def __init__(
            self,
            pad_idx: int,
            nb_blocks: int = 5,
            block_size: int = 5,
            push_to_pad_scale_factor: float = 0.05,
            push_to_pad_label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.nb_blocks = nb_blocks
        self.block_size = block_size
        self.push_to_pad_scale_factor = push_to_pad_scale_factor

        # Loss to push towards occupations, and where loss is
        # invariant towards the order of predicted "blocks"
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            reduction='none',
        )

        # Loss to push towards padding
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
            self.padding_mask.repeat(yhat.size(0), 1), # expand mask to batch size
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
                break # Only padding remains, which we ignore

            block_losses = []

            for candidate_block in range(self.nb_blocks):
                candidate_start_idx = candidate_block * self.block_size
                candidate_end_idx = candidate_start_idx + self.block_size

                block_loss = self.cross_entropy(
                    yhat[:, :, candidate_start_idx:candidate_end_idx],
                    target[:, start_idx:end_idx],
                ).mean(dim=1, keepdim=True)
                block_losses.append(block_loss)

            block_losses = torch.cat(block_losses, dim=1)
            block_loss, _ = block_losses.min(dim=1)
            losses.append(block_loss.mean())

        return sum(losses) / len(losses) # scale to ensure invariant to number of target blocks

    def forward(
            self,
            yhat: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 1, VOCAB]
            target: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 2]
    ) -> Tensor: # pylint: disable=C0116
        '''
        Forward pass for the loss calculation.

        Parameters
        ----------
        yhat : Tensor
            [BATCH_SIZE, `block_size * nb_blocks + 1`, VOCAB_SIZE]-shaped tensor
            consisting of the output from a forward pass from a model. The model
            does not predict the initial BOS-token, but there is space for a final
            EOS token, which explains the `+ 1`-part of the tensor's second dimension.
        target : Tensor
            [BATCH_SIZE, `block_size * nb_blocks + 2`]-shaped tensor consisting
            of the target corresponding to the model output from the given forward
            pass. Includes BOS and EOS tokens, which explain the `+ 2`-part of the
            tensor's second dimension.

        Returns
        -------
        Tensor
            Total loss stored as a 1-element tensor.
        '''
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
        target = target[:, 1:-1] # ignore initial BOS and final EOS

        order_invariant_loss = self._order_invariant_loss(yhat, target)
        push_to_pad_loss = self._push_to_pad(yhat)

        loss = order_invariant_loss + self.push_to_pad_scale_factor * push_to_pad_loss

        return loss


class BlockOrderInvariantLoss(nn.Module):
    '''
    seq2seq-style loss function for sequences consisting of
    `nb_blocks` blocks, all of size `block_size`. The module
    consists of two types of losses:

    1) BLOCK ORDER-INVARIANT CLASSIFICATION
    For each target block, calculate the loss with respect
    to the first k candidate blocks, where k is the number
    of target blocks in the specific observation, i.e., this
    may vary between observations in the same batch. Then
    choose the smallest of the k candidate losses as the loss
    with respect to the given target block. Do so for each
    of the k target blocks and average these to obtain the
    final loss related to the block order-invariant classi-
    fication loss.

    Note that the above procedure is calculated separately
    for each observation in the input batch, i.e., for the
    first target block, two different observations in the same
    batch may end up selecting different candidate blocks.

    Additionally, since most targets consist mainly of empty
    blocks, which are denoted using the `pad_idx` value,
    we ignore its index when calculating the loss. This ensures
    that the loss for the second target block only depends on
    the observations in the batch for which the second target
    block is not empty.

    2) PUSH TOWARDS SPARSITY
    To ensure the model is pushed towards only producing as
    many candidates as there are actual non-empty target blocks,
    apply cross entropy with respect to the blocks with no target
    block, once again at the level of the individual observation.

    Parameters
    ----------
    pad_idx : int
        Which index used to represent padding.
    nb_blocks : int
        Number of blocks. Excluding BOS and EOS tokens, each
        sequence has length `nb_blocks * block_size`.
    block_size : int
        Number of elements in each block. Excluding BOS and EOS
        tokens, each sequence has length `nb_blocks * block_size`.
    push_to_pad_scale_factor : float
        Scaling factor applied to the second part of the loss.
        The total loss consists of the first part of the loss
        added with this times the second part of the loss.
    push_to_pad_label_smoothing : float
        Label smoothing factor applied to the second part of the
        loss.

    '''
    def __init__(
            self,
            pad_idx: int,
            nb_blocks: int = 5,
            block_size: int = 5,
            push_to_pad_scale_factor: float = 1.0,
            push_to_pad_label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.nb_blocks = nb_blocks
        self.block_size = block_size
        self.push_to_pad_scale_factor = push_to_pad_scale_factor

        # Loss to push towards occupations, and where loss is
        # invariant towards the order of predicted "blocks"
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            reduction='none',
        )

        # Loss to push towards padding
        self.padding_cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=push_to_pad_label_smoothing,
            reduction='none',
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
            target_mask: Tensor,
    ) -> Tensor:
        padding_loss = self.padding_cross_entropy(
            yhat,
            self.padding_mask.repeat(yhat.size(0), 1), # expand mask to batch size
        )

        # Block-wise push towards padding
        padding_loss = padding_loss.view(yhat.size(0), self.nb_blocks, self.block_size).mean(dim=2)

        # Only count as loss if no target block, otherwise set to zero
        # to not push towards padding where predictions should occur
        padding_loss[~target_mask] = 0

        return padding_loss.mean()

    def _order_invariant_loss(
            self,
            yhat: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS, VOCAB]
            target: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS]
            target_mask: Tensor,
    ) -> Tensor:
        losses = []

        for target_block in range(self.nb_blocks):
            # Look at target block and calculate loss
            # with respect to all candidate predictions

            start_idx = target_block * self.block_size
            end_idx = start_idx + self.block_size

            if (target[:, start_idx:end_idx] == self.pad_idx).all():
                break # Only padding remains, which we ignore

            block_losses = []

            for candidate_block in range(self.nb_blocks):
                candidate_start_idx = candidate_block * self.block_size
                candidate_end_idx = candidate_start_idx + self.block_size

                block_loss = self.cross_entropy(
                    yhat[:, :, candidate_start_idx:candidate_end_idx],
                    target[:, start_idx:end_idx],
                ).mean(dim=1, keepdim=True)
                block_losses.append(block_loss)

            block_losses = torch.cat(block_losses, dim=1)

            # Disregard block losses from candidate blocks which are
            # beyond number of target blocks
            block_losses[target_mask] = torch.inf

            block_loss, _ = block_losses.min(dim=1)
            losses.append(block_loss.mean())

        return sum(losses) / len(losses) # scale to ensure invariant to number of target blocks

    def _get_target_mask(
            self,
            target: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS]
    ) -> Tensor:
        # Number of target blocks for each observation in batch
        num_target_blocks = torch.argmax((target == self.pad_idx).long(), dim=1) // self.block_size

        # If there are NO padding elements, i.e., number of target
        # blocks equals self.nb_blocks, the above code will wrongly
        # lead to everything being set to True (where it should be
        # False). This occurs as num_target_blocks will only contain
        # zeros for the respective target, meaning the argmax above
        # will lead to a zero. We know that zero should NEVER occur,
        # and we can thus fix this by casting all zeros to the maximum
        # number of target blocks
        num_target_blocks[num_target_blocks == 0] = self.nb_blocks

        # 0:self.nb_blocks-arrangement used to create mask
        arrangement = torch.arange(self.nb_blocks).expand(len(target), -1).to(self.padding_mask.device)

        # Mask to indicate which block losses should be discarded
        target_mask = arrangement >= num_target_blocks.unsqueeze(1)

        return target_mask

    def forward(
            self,
            yhat: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 1, VOCAB]
            target: Tensor, # [BATCH_SIZE, BLOCK_SIZE * NB_BLOCKS + 2]
    ) -> Tensor: # pylint: disable=C0116
        '''
        Forward pass for the loss calculation.

        Parameters
        ----------
        yhat : Tensor
            [BATCH_SIZE, `block_size * nb_blocks + 1`, VOCAB_SIZE]-shaped tensor
            consisting of the output from a forward pass from a model. The model
            does not predict the initial BOS-token, but there is space for a final
            EOS token, which explains the `+ 1`-part of the tensor's second dimension.
        target : Tensor
            [BATCH_SIZE, `block_size * nb_blocks + 2`]-shaped tensor consisting
            of the target corresponding to the model output from the given forward
            pass. Includes BOS and EOS tokens, which explain the `+ 2`-part of the
            tensor's second dimension.

        Returns
        -------
        Tensor
            Total loss stored as a 1-element tensor.
        '''
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
        target = target[:, 1:-1] # ignore initial BOS and final EOS

        # If a target consists of k blocks, only count the first
        # k candidate blocks as relevant prediction and push all
        # other towards padding
        target_mask = self._get_target_mask(target)

        order_invariant_loss = self._order_invariant_loss(yhat, target, target_mask)
        push_to_pad_loss = self._push_to_pad(yhat, target_mask)

        loss = order_invariant_loss + self.push_to_pad_scale_factor * push_to_pad_loss

        return loss
