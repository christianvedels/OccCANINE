import math
import unittest

import torch

import pandas as pd

from histocc import OccCANINE, DATASETS
from histocc.formatter import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX
from histocc.prediction_assets import ModelName, ModelType, SystemType


class AbstractTestWrapperOccCANINE(unittest.TestCase):
    default_params = {
        'device': None,
        'batch_size': 8,
        'verbose': False,
        'baseline': False,
        'force_download': False,
    }
    nodes_to_block = [UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX]

    map_model_type_supported_settings = {
        'flat': {
            'pred_type': ['flat'],
            'behavior_type': ['fast'],
        },
        'seq2seq': {
            'pred_type': ['greedy', 'full'],
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
            hf: bool = False,
            system: SystemType = 'hisco',
            descriptions: pd.DataFrame | None = None,
            use_within_block_sep: bool = False,
            block_nodes: bool = False,
            ):
        params = self.default_params.copy()

        params['system'] = system
        params['descriptions'] = descriptions
        params['use_within_block_sep'] = use_within_block_sep
        params['hf'] = hf

        if model_type is not None:
            # Arg implies untrained model -> skip any loading
            params['model_type'] = model_type
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

    def _run_wrapper_tests(
            self,
            wrapper: OccCANINE,
            sample_inputs: list[str],
            sample_outputs: list[int | str] | None = None,
            ):
        # Supported settings
        supported_settings = self.map_model_type_supported_settings[wrapper.model_type]

        # Test `predict` without args (not supported for `flat` model type)
        if wrapper.model_type != 'flat':
            with self.subTest(msg='No arguments prediction'):
                pred = wrapper.predict(sample_inputs)
                self.assertEqual(len(pred), len(sample_inputs))

            if sample_outputs is not None:
                with self.subTest(msg='No arguments prediction, verifying results'):
                    self.assertListEqual(
                        list(pred[f'{wrapper.system}_1']),
                        sample_outputs,
                    )

        for prediction_type in supported_settings['pred_type']:
            for behavior in supported_settings['behavior_type']:
                with self.subTest(prediction_type=prediction_type):
                    # Test standard prediction
                    pred = wrapper.predict(
                        sample_inputs,
                        prediction_type=prediction_type,
                        behavior=behavior,
                        what='pred',
                        )
                    self.assertEqual(len(pred), len(sample_inputs))

                if sample_outputs is not None:
                    with self.subTest(msg=f'prediction_type={prediction_type}, verifying results'):
                        self.assertListEqual(
                            list(pred[f'{wrapper.system}_1']),
                            sample_outputs,
                        )

                # Test probability if prediction type is not greedy
                if prediction_type == 'greedy':
                    continue

                with self.subTest(prediction_type=prediction_type):
                    pred = wrapper.predict(
                        sample_inputs,
                        prediction_type=prediction_type,
                        behavior=behavior,
                        what='probs',
                        )
                    self.assertEqual(len(pred), len(sample_inputs))


class TestWrapperHISCOOccCANINE(AbstractTestWrapperOccCANINE):
    model_names = ['OccCANINE', 'OccCANINE_s2s', 'OccCANINE_s2s_mix']
    sample_inputs = ['he is a farmer', 'he is a fisherman']
    sample_outputs = ['61110', '64100']

    def test_wrappers(self):
        # Run tests for untrained seq2seq wrapper
        wrapper_s2s = self._initialize_model(
            model_type='seq2seq',
            system='hisco',
            block_nodes=True,
            )
        self._run_wrapper_tests(wrapper_s2s, self.sample_inputs)
        del wrapper_s2s

        # Run tests for untrained mixer wrapper
        wrapper_mixer = self._initialize_model(
            model_type='mix',
            system='hisco',
            block_nodes=True,
            )
        self._run_wrapper_tests(wrapper_mixer, self.sample_inputs)
        del wrapper_mixer

        # Run tests for pretrained models; here, we also check if predictions
        # match the labels associated with our sample inputs
        for model_name in self.model_names:
            wrapper_pretrained = self._initialize_model(
                name=model_name,
                hf=True,
                system='hisco',
            )
            self._run_wrapper_tests(
                wrapper=wrapper_pretrained,
                sample_inputs=self.sample_inputs,
                sample_outputs=self.sample_outputs,
            )
            del wrapper_pretrained


class TestWrapperGeneralPurposeOccCANINE(AbstractTestWrapperOccCANINE):
    sample_inputs = ['he is a farmer', 'he is a fisherman']

    def setUp(self):
        super().setUp()

    def test_wrapper_occ1950(self):
        wrapper_mixer_occ1950 = self._initialize_model(
            name=r'Y:\pc-to-Y\hisco\ft-exp\250408\mixer-occ1950-ft\last.bin',
            system='occ1950',
            )
        self._run_wrapper_tests(
            wrapper_mixer_occ1950,
            self.sample_inputs,
            sample_outputs=['100', '910'],
            )
        del wrapper_mixer_occ1950

    def test_wrapper_icem(self):
        wrapper_mixer_icem = self._initialize_model(
            name=r'Y:\pc-to-Y\hisco\ft-exp\250408\mixer-icem-ft\last.bin',
            system='icem',
            )
        self._run_wrapper_tests(
            wrapper_mixer_icem,
            self.sample_inputs,
            sample_outputs=['173', '194'],
            )
        del wrapper_mixer_icem

    def test_wrapper_psti(self):
        wrapper_mixer_psti = self._initialize_model(
            name=r'Y:\pc-to-Y\hisco\ft-exp\250408\mixer-psti-ft\last.bin',
            system='psti',
            use_within_block_sep=True,
            )
        self._run_wrapper_tests(
            wrapper_mixer_psti,
            self.sample_inputs,
            sample_outputs=['1,1,0,0,3,0,0,0', '1,5,0,0,0,0,1,0'],
            )
        del wrapper_mixer_psti

    def test_wrapper_isco(self):
        wrapper_mixer_isco = self._initialize_model(
            name=r'Y:\pc-to-Y\hisco\ft-exp\250408\mixer-isco-ft\last.bin',
            system='isco',
            )
        self._run_wrapper_tests(
            wrapper_mixer_isco,
            self.sample_inputs,
            sample_outputs=['611', '641'],
            )
        del wrapper_mixer_isco


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
