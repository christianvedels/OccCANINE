import random
import unittest

import numpy as np
import pandas as pd

from histocc import DATASETS
from histocc.formatter import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    BlockyFormatter,
    construct_general_purpose_formatter,
)


class TestBlockyFormatter(unittest.TestCase):
    def setUp(self):
        self.toydata = DATASETS['toydata']()

    def _test_transform_label(
            self,
            raw_input: str | pd.DataFrame | pd.Series,
            expected_output: np.ndarray,
            formatter: BlockyFormatter,
    ):
        with self.subTest(msg=f'transform_label test for raw_input={raw_input}, formatter={formatter}'):
            formatted = formatter.transform_label(raw_input)
            np.testing.assert_array_equal(formatted, expected_output)

    def _test_clean_pred(
            self,
            pred: np.ndarray,
            expected_cleaned: str,
            formatter: BlockyFormatter,
    ):
        with self.subTest(msg=f'clean_pred test for pred={pred}, formatter={formatter}'):
            clean = formatter.clean_pred(pred)
            self.assertEqual(clean, expected_cleaned)

    def _test_cycle_consistency(
            self,
            raw_input: str,
            formatter: BlockyFormatter,
    ):
        formatted = formatter.transform_label(raw_input)
        re_transformed = formatter.clean_pred(formatted.astype(int))
        self.assertEqual(raw_input, re_transformed)

    def run_cycle_consistency_tests(
            self,
            formatter: BlockyFormatter,
            chars: list[str | int],
            within_block_sep: str = '',
            runs: int = 10_000,
    ):
        # Test cycle consistency for random string inputs. Exhaustive
        # check not feasible
        for _ in range(runs):
            raw_input = []

            for _ in range(random.randint(1, formatter.max_num_codes)):
                num_tokens = random.randint(1, formatter.block_size)

                code = random.sample(chars, num_tokens) # this selects unique elements, which is not an actual requirement
                code = [str(c) for c in code]
                code_joined = within_block_sep.join(code)

                raw_input.append(code_joined)

            raw_input = formatter.sep_value.join(raw_input)

            with self.subTest(msg=f'cycle-consistency check for raw_input={raw_input}, formatter={formatter}'):
                self._test_cycle_consistency(raw_input, formatter)

    def test_with_within_block_sep(self):
        within_block_sep = ','

        formatter = construct_general_purpose_formatter(
            block_size=3,
            target_cols=[0, 0],
            use_within_block_sep=True,
        )
        sep_value = formatter.sep_value

        # Transform label
        self._test_transform_label(
            raw_input=f'1{within_block_sep}2{within_block_sep}3{sep_value}1{within_block_sep}1{within_block_sep}1',
            expected_output=np.array([
                BOS_IDX, 1005, 1006, 1007, 1005, 1005, 1005, EOS_IDX
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'1{within_block_sep}2{within_block_sep}3{sep_value}1{within_block_sep}1',
            expected_output=np.array([
                BOS_IDX, 1005, 1006, 1007, 1005, 1005, EOS_IDX, EOS_IDX
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'1{within_block_sep}2{within_block_sep}3',
            expected_output=np.array([
                BOS_IDX, 1005, 1006, 1007, PAD_IDX, PAD_IDX, PAD_IDX, EOS_IDX
                ]),
            formatter=formatter,
        )

        # Clean prediction
        self._test_clean_pred(
            pred=np.array([BOS_IDX, 1005, 1006, 1007, PAD_IDX, PAD_IDX, PAD_IDX, EOS_IDX]),
            expected_cleaned=f'1{within_block_sep}2{within_block_sep}3',
            formatter=formatter,
        )
        self._test_clean_pred(
            pred=np.array([BOS_IDX, 6, 1006, 1007, 5, 5, 5, EOS_IDX]),
            expected_cleaned=f'-998{within_block_sep}2{within_block_sep}3{sep_value}-999{within_block_sep}-999{within_block_sep}-999',
            formatter=formatter,
        )
        self._test_clean_pred(
            pred=np.array([BOS_IDX, 2030, 2005, 2029, 2064, EOS_IDX, EOS_IDX, EOS_IDX]),
            expected_cleaned=f'A{within_block_sep}b{within_block_sep}z{sep_value}*',
            formatter=formatter,
        )

        # Cycle consistency
        self.run_cycle_consistency_tests(
            formatter=formatter,
            chars=sorted(formatter.map_char_idx.keys()),
            within_block_sep=within_block_sep,
        )

    def test_without_within_block_sep(self):
        formatter = construct_general_purpose_formatter(
            block_size=3,
            target_cols=[0, 0],
            use_within_block_sep=False,
        )
        sep_value = formatter.sep_value

        # Only want 1-char tokens since no separator
        chars = [str(x) for x in formatter.map_char_idx.keys() if len(str(x)) == 1]

        # Transform label
        self._test_transform_label(
            raw_input=f'123{sep_value}111',
            expected_output=np.array([
                BOS_IDX, 1005, 1006, 1007, 1005, 1005, 1005, EOS_IDX
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'123{sep_value}11',
            expected_output=np.array([
                BOS_IDX, 1005, 1006, 1007, 1005, 1005, EOS_IDX, EOS_IDX
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input='123',
            expected_output=np.array([
                BOS_IDX, 1005, 1006, 1007, PAD_IDX, PAD_IDX, PAD_IDX, EOS_IDX
                ]),
            formatter=formatter,
        )

        # Clean prediction
        self._test_clean_pred(
            pred=np.array([BOS_IDX, 1005, 1006, 1007, PAD_IDX, PAD_IDX, PAD_IDX, EOS_IDX]),
            expected_cleaned='123',
            formatter=formatter,
        )
        self._test_clean_pred(
            pred=np.array([BOS_IDX, 2030, 2005, 2029, 2064, EOS_IDX, EOS_IDX, EOS_IDX]),
            expected_cleaned=f'Abz{sep_value}*',
            formatter=formatter,
        )

        # Cycle consistency
        self.run_cycle_consistency_tests(
            formatter=formatter,
            chars=chars,
        )

    def test_toydata(self):
        formatter = construct_general_purpose_formatter(
            block_size=5,
            target_cols=['hisco_1'],
        )
        for i in range(len(self.toydata)):
            transformed = formatter.transform_label(self.toydata.iloc[i])
            cleaned = formatter.clean_pred(transformed.astype(int))

            expected = self.toydata.iloc[i]['hisco_1']
            expected = str(expected)

            self.assertEqual(cleaned, expected)

    def test_psti_formatter(self):
        formatter = construct_general_purpose_formatter(
            block_size=8,
            target_cols=[0, 0],
            use_within_block_sep=True,
        )
        sep_value = formatter.sep_value

        # Transform label
        self._test_transform_label(
            raw_input='2,2,2,0,2,6,1,0',
            expected_output=np.array([
                BOS_IDX,
                1006, 1006, 1006, 1004, 1006, 1010, 1005, 1004,
                PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'2,2,2,0,2,6,1,0{sep_value}2,2,1,0,17,2,1,0',
            expected_output=np.array([
                BOS_IDX,
                1006, 1006, 1006, 1004, 1006, 1010, 1005, 1004,
                1006, 1006, 1005, 1004, 1021, 1006, 1005, 1004,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        # Clean prediction
        self._test_clean_pred(
            pred=np.array([
                BOS_IDX,
                1007, 1010, 1005, 1004, 1005, 1005, 1004, 1004,
                1007, 1009, 1004, 1004, 1005, 1008, 1004, 1004,
                EOS_IDX,
                ]),
            expected_cleaned=f'3,6,1,0,1,1,0,0{sep_value}3,5,0,0,1,4,0,0',
            formatter=formatter,
        )

    def test_isco_formatter(self):
        formatter = construct_general_purpose_formatter(
            block_size=3,
            target_cols=[0, 0],
        )
        sep_value = formatter.sep_value

        # Transform label
        self._test_transform_label(
            raw_input='999',
            expected_output=np.array([
                BOS_IDX,
                1013, 1013, 1013,
                PAD_IDX, PAD_IDX, PAD_IDX,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'540{sep_value}711',
            expected_output=np.array([
                BOS_IDX,
                1009, 1008, 1004,
                1011, 1005, 1005,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        # Clean prediction
        self._test_clean_pred(
            pred=np.array([
                BOS_IDX,
                1010, 1006, 1008,
                1005, 1005, EOS_IDX,
                EOS_IDX,
                ]),
            expected_cleaned=f'624{sep_value}11',
            formatter=formatter,
        )

    def test_icem_formatter(self):
        formatter = construct_general_purpose_formatter(
            block_size=3,
            target_cols=[0, 0, 0],
        )
        sep_value = formatter.sep_value

        # Transform label
        self._test_transform_label(
            raw_input='787',
            expected_output=np.array([
                BOS_IDX,
                1011, 1012, 1011,
                PAD_IDX, PAD_IDX, PAD_IDX,
                PAD_IDX, PAD_IDX, PAD_IDX,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'778{sep_value}84{sep_value}?',
            expected_output=np.array([
                BOS_IDX,
                1011, 1011, 1012,
                1012, 1008, EOS_IDX,
                2074, EOS_IDX, EOS_IDX,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        # Clean prediction
        self._test_clean_pred(
            pred=np.array([
                BOS_IDX,
                1010, 1013, 1011,
                1005, EOS_IDX, EOS_IDX,
                PAD_IDX, PAD_IDX, PAD_IDX,
                EOS_IDX,
                ]),
            expected_cleaned=f'697{sep_value}1',
            formatter=formatter,
        )

    def test_occ1950_formatter(self):
        formatter = construct_general_purpose_formatter(
            block_size=3,
            target_cols=[0, 0, 0],
        )
        sep_value = formatter.sep_value

        # Transform label
        self._test_transform_label(
            raw_input='787',
            expected_output=np.array([
                BOS_IDX,
                1011, 1012, 1011,
                PAD_IDX, PAD_IDX, PAD_IDX,
                PAD_IDX, PAD_IDX, PAD_IDX,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        self._test_transform_label(
            raw_input=f'778{sep_value}84{sep_value}?',
            expected_output=np.array([
                BOS_IDX,
                1011, 1011, 1012,
                1012, 1008, EOS_IDX,
                2074, EOS_IDX, EOS_IDX,
                EOS_IDX,
                ]),
            formatter=formatter,
        )
        # Clean prediction
        self._test_clean_pred(
            pred=np.array([
                BOS_IDX,
                1010, 1013, 1011,
                1005, EOS_IDX, EOS_IDX,
                PAD_IDX, PAD_IDX, PAD_IDX,
                EOS_IDX,
                ]),
            expected_cleaned=f'697{sep_value}1',
            formatter=formatter,
        )

    # def test_hisco_formatter(self):
    #     raise NotImplementedError


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
