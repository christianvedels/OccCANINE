import unittest

import torch

from torch import Tensor

from histocc import OrderInvariantSeq2SeqCrossEntropy, BlockOrderInvariantLoss
from histocc.formatter import PAD_IDX, BOS_IDX, EOS_IDX, hisco_blocky5


def _gen_perfect_pred(target: Tensor, vocab_size: int, strength: int | float = 1000) -> Tensor:
    # First make only-zeroes pred with shape [BATCH_SIZE = 2, SEQ LEN = 26, VOCAB_SIZE = 18]
    yhat = torch.zeros(
        size=(target.size(0), target.size(1) - 1, vocab_size),
        )

    # Next fill in values to match target
    for batch_idx, batch in enumerate(target[:, 1:]): # We do not predict initial BOS_IDX
        for seq_idx, elem in enumerate(batch):
            yhat[batch_idx, seq_idx, elem] = strength

    # Test prediction is perfect
    max_index = torch.argmax(yhat, dim=2)
    assert (max_index == target[:, 1:]).all()

    return yhat


def _gen_pure_padding_pred(target: Tensor, vocab_size: int, strength: int | float = 1000) -> Tensor:
    # First make only-zeroes pred with shape [BATCH_SIZE = 2, SEQ LEN = 26, VOCAB_SIZE = 18]
    yhat = torch.zeros(
        size=(target.size(0), target.size(1) - 1, vocab_size),
        )

    # Next fill in values to match target
    for batch_idx, batch in enumerate(target[:, 1:]): # We do not predict initial BOS_IDX
        for seq_idx in range(len(batch)):
            yhat[batch_idx, seq_idx, PAD_IDX] = strength

    # Test prediction is perfect
    max_index = torch.argmax(yhat, dim=2)
    assert (max_index == PAD_IDX).all()

    return yhat


class TestOrderInvariantSeq2SeqCrossEntropy(unittest.TestCase):
    baseline_target = torch.tensor([
        [
            BOS_IDX,
            10, 11, 12, 13, 14, # 10 = 2, 11 = 3, etc
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            EOS_IDX,
        ],
        [
            BOS_IDX,
            7, 7, 7, 7, 7, # 7 = -1
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            EOS_IDX,
        ],
    ], dtype=torch.long)
    padding_target = torch.ones(
        size=(2, 5 * 5 + 2),
        dtype=torch.long,
        ) * PAD_IDX

    def setUp(self):
        self.formatter = hisco_blocky5()
        self.loss = OrderInvariantSeq2SeqCrossEntropy(
            pad_idx=PAD_IDX,
            nb_blocks=self.formatter.max_num_codes,
            block_size=5, # A HISCO code consists of 5 digits (we treat -1, -2, and -3 this way)
        )

        self.perfect_baseline_pred = _gen_perfect_pred(
            self.baseline_target,
            vocab_size=max(self.formatter.map_idx_char) + 1,
        )
        self.pure_padding_pred = _gen_pure_padding_pred(
            self.padding_target,
            vocab_size=max(self.formatter.map_idx_char) + 1,
        )

    def _get_order_invariant_loss(self, yhat: Tensor, target: Tensor) -> Tensor:
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
        target = target[:, 1:-1] # Skip initial BOS, final EOS

        return self.loss._order_invariant_loss(yhat, target) # pylint: disable=W0212

    def _get_push_to_pad_loss(self, yhat: Tensor) -> Tensor:
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]

        return self.loss._push_to_pad(yhat) # pylint: disable=W0212

    def test_aligned(self):
        order_invariant_loss = self._get_order_invariant_loss(
            self.perfect_baseline_pred,
            self.baseline_target,
            )
        self.assertAlmostEqual(order_invariant_loss.item(), 0)

    def _test_aligned_with_shifts(
            self,
            target: Tensor,
            perfect_pred: Tensor,
            ):
        for i in range(self.formatter.max_num_codes - 1):
            start_idx_i = i * 5
            end_idx_i = start_idx_i + 5

            for j in range(i + 1, self.formatter.max_num_codes):
                start_idx_j = j * 5
                end_idx_j = start_idx_j + 5

                pred = perfect_pred.clone()
                pred[:, start_idx_i:end_idx_i, :] = self.perfect_baseline_pred[:, start_idx_j:end_idx_j, :]
                pred[:, start_idx_j:end_idx_j, :] = self.perfect_baseline_pred[:, start_idx_i:end_idx_i, :]

                order_invariant_loss = self._get_order_invariant_loss(
                    pred,
                    target,
                    )

                self.assertAlmostEqual(order_invariant_loss.item(), 0)

    def test_aligned_with_shifts(self):
        self._test_aligned_with_shifts(self.baseline_target, self.perfect_baseline_pred)

    def _test_aligned_with_shifts_elem_in_batch_wise(
            self,
            target: Tensor,
            perfect_pred: Tensor,
            ):
        for batch_idx in range(len(target)):
            for i in range(self.formatter.max_num_codes - 1):
                start_idx_i = i * 5
                end_idx_i = start_idx_i + 5

                for j in range(i + 1, self.formatter.max_num_codes):
                    start_idx_j = j * 5
                    end_idx_j = start_idx_j + 5

                    pred = perfect_pred.clone()
                    pred[batch_idx, start_idx_i:end_idx_i, :] = self.perfect_baseline_pred[batch_idx, start_idx_j:end_idx_j, :]
                    pred[batch_idx, start_idx_j:end_idx_j, :] = self.perfect_baseline_pred[batch_idx, start_idx_i:end_idx_i, :]

                    order_invariant_loss = self._get_order_invariant_loss(
                        pred,
                        target,
                        )

                    self.assertAlmostEqual(order_invariant_loss.item(), 0)

    def test_aligned_with_shifts_elem_in_batch_wise(self):
        self._test_aligned_with_shifts_elem_in_batch_wise(self.baseline_target, self.perfect_baseline_pred)

    def test_padding_loss(self):
        push_to_padding_loss = self._get_push_to_pad_loss(self.pure_padding_pred)
        self.assertAlmostEqual(push_to_padding_loss.item(), 0)

    def test_aligned_full_loss(self):
        full_loss = self.loss(self.perfect_baseline_pred, self.baseline_target)

        # A "perfect" prediction with at least one HISCO code present will
        # have non-zero loss due to push-to-pad mechanism
        self.assertGreater(full_loss.item(), self.loss.push_to_pad_scale_factor) # pylint: disable=W0212


class TestBlockOrderInvariantLoss(unittest.TestCase):
    baseline_target = torch.tensor([
        [
            BOS_IDX,
            10, 11, 12, 13, 14, # 10 = 2, 11 = 3, etc
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            EOS_IDX,
        ],
        [
            BOS_IDX,
            7, 7, 7, 7, 7, # 7 = -1
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            *([PAD_IDX] * 5),
            EOS_IDX,
        ],
    ], dtype=torch.long)
    padding_target = torch.ones(
        size=(2, 5 * 5 + 2),
        dtype=torch.long,
        ) * PAD_IDX

    def setUp(self):
        self.formatter = hisco_blocky5()
        self.loss = BlockOrderInvariantLoss(
            pad_idx=PAD_IDX,
            nb_blocks=self.formatter.max_num_codes,
            block_size=5, # A HISCO code consists of 5 digits (we treat -1, -2, and -3 this way)
        )

        self.perfect_baseline_pred = _gen_perfect_pred(
            self.baseline_target,
            vocab_size=max(self.formatter.map_idx_char) + 1,
        )
        self.pure_padding_pred = _gen_pure_padding_pred(
            self.padding_target,
            vocab_size=max(self.formatter.map_idx_char) + 1,
        )

    def _get_order_invariant_loss(self, yhat: Tensor, target: Tensor) -> Tensor:
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
        target = target[:, 1:-1] # Skip initial BOS, final EOS
        target_mask = self.loss._get_target_mask(target) # pylint: disable=W0212

        return self.loss._order_invariant_loss(yhat, target, target_mask) # pylint: disable=W0212

    def _get_push_to_pad_loss(self, yhat: Tensor, target: Tensor) -> Tensor:
        yhat = yhat.permute(0, 2, 1)[:, :, :-1] # -> [BATCH_SIZE, VOCAB, BLOCK_SIZE * NB_BLOCKS]
        target_mask = self.loss._get_target_mask(target) # pylint: disable=W0212

        return self.loss._push_to_pad(yhat, target_mask) # pylint: disable=W0212

    def test_aligned(self):
        order_invariant_loss = self._get_order_invariant_loss(
            self.perfect_baseline_pred,
            self.baseline_target,
            )
        self.assertAlmostEqual(order_invariant_loss.item(), 0)

    def _test_aligned_with_shifts(
            self,
            target: Tensor,
            perfect_pred: Tensor,
            ):
        for i in range(self.formatter.max_num_codes - 1):
            start_idx_i = i * 5
            end_idx_i = start_idx_i + 5

            for j in range(i + 1, self.formatter.max_num_codes):
                start_idx_j = j * 5
                end_idx_j = start_idx_j + 5

                pred = perfect_pred.clone()
                pred[:, start_idx_i:end_idx_i, :] = self.perfect_baseline_pred[:, start_idx_j:end_idx_j, :]
                pred[:, start_idx_j:end_idx_j, :] = self.perfect_baseline_pred[:, start_idx_i:end_idx_i, :]

                order_invariant_loss = self._get_order_invariant_loss(
                    pred,
                    target,
                    )

                self.assertAlmostEqual(order_invariant_loss.item(), 0)

    def test_aligned_with_shifts(self):
        self._test_aligned_with_shifts(self.baseline_target, self.perfect_baseline_pred)

    def _test_aligned_with_shifts_elem_in_batch_wise(
            self,
            target: Tensor,
            perfect_pred: Tensor,
            ):
        # FIXME we now can no longer just shift, but we can reverse, so implement
        # that instead
        return

        for batch_idx in range(len(target)):
            for i in range(self.formatter.max_num_codes - 1):
                start_idx_i = i * 5
                end_idx_i = start_idx_i + 5

                for j in range(i + 1, self.formatter.max_num_codes):
                    start_idx_j = j * 5
                    end_idx_j = start_idx_j + 5

                    pred = perfect_pred.clone()
                    pred[batch_idx, start_idx_i:end_idx_i, :] = self.perfect_baseline_pred[batch_idx, start_idx_j:end_idx_j, :]
                    pred[batch_idx, start_idx_j:end_idx_j, :] = self.perfect_baseline_pred[batch_idx, start_idx_i:end_idx_i, :]

                    order_invariant_loss = self._get_order_invariant_loss(
                        pred,
                        target,
                        )

                    self.assertAlmostEqual(order_invariant_loss.item(), 0)

    def test_aligned_with_shifts_elem_in_batch_wise(self):
        self._test_aligned_with_shifts_elem_in_batch_wise(self.baseline_target, self.perfect_baseline_pred)

    def test_padding_loss(self):
        # FIXME we need target to call self._get_push_to_pad_loss now
        pass
        # push_to_padding_loss = self._get_push_to_pad_loss(self.pure_padding_pred)
        # self.assertAlmostEqual(push_to_padding_loss.item(), 0)

    def test_aligned_full_loss(self):
        full_loss = self.loss(self.perfect_baseline_pred, self.baseline_target)

        # A "perfect" prediction with at least one HISCO code present will
        # have non-zero loss due to push-to-pad mechanism
        self.assertAlmostEqual(full_loss.item(), 0) # pylint: disable=W0212


if __name__ == '__main__':
    unittest.main()
