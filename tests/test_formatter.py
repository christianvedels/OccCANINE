import unittest

import numpy as np
import pandas as pd

from histocc.formatter import blocky5


class TestBlockyHISCOFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = blocky5()

    def test_transform_label_string_input(self):
        raw_input = '12345&-1'
        expected_output = np.array([
            2., 9., 10., 11., 12., 13., 7., 7., 7., 7., 7., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.,
            ])

        result = self.formatter.transform_label(raw_input)

        np.testing.assert_array_equal(result, expected_output)

    def test_transform_label_dataframe_input(self):
        data = {'code1': [123], 'code2': [0], 'code3': [2], 'code4': [None], 'code5': [None]}
        raw_input = pd.DataFrame(data)

        expected_output = np.array([
            2.,  8., 11., 13., 13.,  8.,  5.,  5.,  5.,  5.,  5.,  7.,  7., 7.,  7.,  7.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 3.,
            ])

        result = self.formatter.transform_label(raw_input)

        np.testing.assert_array_equal(result, expected_output)

    def test_transform_label_none_input(self):
        raw_input = None
        result = self.formatter.transform_label(raw_input)
        self.assertIsNone(result)

    # We currently do not support empty strings, but perhaps we should
    # def test_transform_label_empty_string_input(self):
    #     raw_input = ''
    #     expected_output = np.array([2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.])

    #     result = self.formatter.transform_label(raw_input)

    #     np.testing.assert_array_equal(result, expected_output)

    def test_clean_pred_valid_input(self):
        raw_pred = np.array([
            2., 9., 10., 11., 12., 13., 9., 10., 11., 12., 13., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.,
        ]).astype(int)
        expected_output = '12345&12345'

        result = self.formatter.clean_pred(raw_pred)

        self.assertEqual(result, expected_output)

    def test_clean_pred_special_codes(self):
        raw_pred = np.array([
            2.,  7.,  7.,  7.,  7.,  7.,  5.,  5.,  5.,  5.,  5.,  9., 10., 11., 12., 13.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 3.
        ]).astype(int)
        expected_output = '-1&-3&12345'

        result = self.formatter.clean_pred(raw_pred)

        self.assertEqual(result, expected_output)

    def test_clean_messy_pred(self):
        raw_pred = np.array([
            2,
            1,  1,  1,  1,  1,
            13, 16, 11, 12, 8,
            1,  1,  1,  1,  8, # messy, but we sort 8 away gracefully as in chunk with PAD_IDX
            14, 9,  9,  9,  13,
            13, 1,  1,  1,  1, # messy, but we sort 13 away gracefully as in chunk with PAD_IDX
            1,
        ]).astype(int)
        expected_output = '58340&61115'

        result = self.formatter.clean_pred(raw_pred)

        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
