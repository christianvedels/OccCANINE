import math
import unittest
import torch
from histocc import OccCANINE, DATASETS
from histocc.formatter import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX
from histocc.prediction_assets import ModelName, ModelType


class TestWrapperOccCANINE(unittest.TestCase):
    default_params = {
        'device': None,
        'batch_size': 8,
        'verbose': False,
        'baseline': False,
        'force_download': False,
    }
    sample_inputs = ['he is a farmer', 'he is a fisherman']
    nodes_to_block = [UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX]

    map_model_type_supported_settings = {
        'flat': {
            'pred_type': ['flat'],
            'behavior_type': ['fast'],
        },
        'seq2seq': {
            'pred_type': ['greedy'],
            'behavior_type': ['good'],
        },
        'mix': {
            'pred_type': ['flat', 'greedy', 'full'],
            'behavior_type': ['fast', 'good'],
        },
    }

    def setUp(self):
        self.toydata = DATASETS['toydata']()

    def _initialize_model(
            self,
            model_type: ModelType | None = None,
            name: ModelName | None = None,
            block_nodes: bool = False,
            ):
        params = self.default_params.copy()

        if model_type is not None:
            # Arg implies untrained model -> skip any loading
            params['model_type'] = model_type
            params['hf'] = False
            params['skip_load'] = True

        if name is not None:
            params['name'] = name

        model_wrapper = OccCANINE(**params)

        # Optionally block specific nodes
        if block_nodes:
            with torch.no_grad():
                bias = model_wrapper.model.decoder.head.bias
                for val in self.nodes_to_block + [model_wrapper.model.decoder.head.out_features - 1]:
                    bias[val] = -math.inf

        return model_wrapper

    def _run_wrapper_tests(self, wrapper: OccCANINE):
        # Supported settings
        supported_settings = self.map_model_type_supported_settings[wrapper.model_type]

        # Test `predict` without args (not supported for `flat` model type)
        if wrapper.model_type != 'flat':
            with self.subTest(msg='No arguments prediction'):
                pred = wrapper.predict(self.sample_inputs)
                self.assertEqual(len(pred), len(self.sample_inputs))

        for prediction_type in supported_settings['pred_type']:
            for behavior in supported_settings['behavior_type']:
                with self.subTest(prediction_type=prediction_type):
                    # Test standard prediction
                    pred = wrapper.predict(
                        self.sample_inputs,
                        prediction_type=prediction_type,
                        behavior=behavior,
                        what='pred',
                        )
                    self.assertEqual(len(pred), len(self.sample_inputs))

                    # Test probability if prediction type is not greedy
                    if prediction_type == 'greedy':
                        continue

                    pred = wrapper.predict(
                        self.sample_inputs,
                        prediction_type=prediction_type,
                        behavior=behavior,
                        what='probs',
                        )
                    self.assertEqual(len(pred), len(self.sample_inputs))

    def test_wrappers(self):
        # Run tests for untrained seq2seq wrapper
        wrapper_s2s = self._initialize_model(
            model_type='seq2seq',
            block_nodes=True,
            )
        self._run_wrapper_tests(wrapper_s2s)
        del wrapper_s2s

        # Run tests for untrained mixer wrapper
        wrapper_mixer = self._initialize_model(
            model_type='mix',
            block_nodes=True,
            )
        self._run_wrapper_tests(wrapper_mixer)
        del wrapper_mixer

        # Run tests for pretrained models
        for model_name in ModelName.__args__:
            with self.subTest(model_name=model_name):
                wrapper_pretrained = self._initialize_model(name=model_name)
                self._run_wrapper_tests(wrapper_pretrained)
                del wrapper_pretrained


class SubtestCountingTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtests_passed = 0
        self.subtests_run = 0

    def addSubTest(self, test, subtest, outcome=None):
        super().addSubTest(test, subtest, outcome)
        self.subtests_run += 1
        if outcome is None:  # Subtest passed
            self.subtests_passed += 1

    def stopTestRun(self):
        super().stopTestRun()
        self.stream.writeln(f"Subtests passed: {self.subtests_passed} of {self.subtests_run}")


class SubtestCountingTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return SubtestCountingTestResult(self.stream, self.descriptions, self.verbosity)


if __name__ == '__main__':
    unittest.main(testRunner=SubtestCountingTestRunner())
