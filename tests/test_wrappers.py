import math
import unittest
import warnings

import torch

from histocc import OccCANINE, DATASETS
from histocc.formatter import (
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX
)


class TestWrapperSeq2SeqOccCANINE(unittest.TestCase):
    default_params = {
        'device': None,
        'batch_size': 8,
        'verbose': False,
        'baseline': False,
        'hf': False,
        'force_download': False,
        'skip_load': True,
    }
    sample_inputs = ['he is a farmer', 'he is a fisherman']

    def setUp(self):
        self.toydata = DATASETS['toydata']()

        self.wrapper_s2s = OccCANINE(
            model_type='seq2seq',
            **self.default_params,
            )
        self.wrapper_mixer = OccCANINE(
            model_type='mix',
            **self.default_params,
            )
        # TODO expand beyond HISCO formatter

        self.wrappers = [
            self.wrapper_s2s,
            self.wrapper_mixer,
        ]

        # Current issue that models are initialized with one
        # additional output node than is supported by our
        # formatter. This may lead to predictions that cannot
        # be handled by our formatter. Therefore, make sure these
        # predictions are never produced
        # Further, formatter does not gracefully handle all special
        # tokens; also block those from predictions
        # These issues occur as we do not use a pre-trained model;
        # the trained models handle this by never producing such
        # predictions
        nodes_to_block = [
            self.wrapper_s2s.model.decoder.head.out_features - 1,
            UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX
        ]

        with torch.no_grad():
            for wrapper in self.wrappers:
                for val in nodes_to_block:
                    wrapper.model.decoder.head.bias[val] = - math.inf

    def test_seq2seq_predict_no_args(self):
        pred = self.wrapper_s2s.predict(
            self.sample_inputs,
            )

        self.assertEqual(len(pred), len(self.sample_inputs))

    def test_mixer_predict_no_args(self):
        pred = self.wrapper_mixer.predict(
            self.sample_inputs,
            )

        self.assertEqual(len(pred), len(self.sample_inputs))

    def _test_predict(
            self,
            wrapper: OccCANINE,
            inputs: list[str],
            behavior: str,
            prediction_type: str | None,
            what: str,
    ):
        out = wrapper.predict(
            inputs,
            behavior=behavior,
            prediction_type=prediction_type,
            what=what,
            )

        self.assertEqual(len(out), len(inputs))

    # Varying prediction_type -- what == pred

    # def test_seq2seq_flat_pred(self):
    #     self._test_predict(
    #         wrapper=self.wrapper_s2s,
    #         inputs=self.sample_inputs,
    #         behavior='good',
    #         prediction_type='flat',
    #         what='pred',
    #     )

    def test_seq2seq_greedy_pred(self):
        self._test_predict(
            wrapper=self.wrapper_s2s,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='greedy',
            what='pred',
        )

    def test_seq2seq_full_pred(self):
        self._test_predict(
            wrapper=self.wrapper_s2s,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='full',
            what='pred',
        )

    def test_mixer_flat_pred(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='flat',
            what='pred',
        )

    def test_mixer_greedy_pred(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='greedy',
            what='pred',
        )

    def test_mixer_full_pred(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='full',
            what='pred',
        )

    # Varying prediction_type -- what == prob
    def test_seq2seq_full_prob(self):
        self._test_predict(
            wrapper=self.wrapper_s2s,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='full',
            what='probs',
        )

    def test_mixer_full_prob(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='full',
            what='probs',
        )

    def test_mixer_flat_prob(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type='flat',
            what='probs',
        )

    # TODO probs but for other prediction_type, wrapper, behavior

    # Varying behavior -- what == pred

    # def test_seq2seq_fast_pred(self):
    #     self._test_predict(
    #         wrapper=self.wrapper_s2s,
    #         inputs=self.sample_inputs,
    #         behavior='fast',
    #         prediction_type=None,
    #         what='pred',
    #     )

    def test_seq2seq_good_pred(self):
        self._test_predict(
            wrapper=self.wrapper_s2s,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type=None,
            what='pred',
        )

    def test_mixer_fast_pred(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='fast',
            prediction_type=None,
            what='pred',
        )

    def test_mixer_good_pred(self):
        self._test_predict(
            wrapper=self.wrapper_mixer,
            inputs=self.sample_inputs,
            behavior='good',
            prediction_type=None,
            what='pred',
        )


if __name__ == '__main__':
    unittest.main()
