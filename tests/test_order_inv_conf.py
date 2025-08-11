import unittest

from histocc import OccCANINE, DATASETS


class AbstractTestWrapperOccCANINE(unittest.TestCase):
    def setUp(self):
        self.toydata = DATASETS['toydata']()
        self.model = OccCANINE()

    def test_probs_higher(self):
        ''' Test estimated probabilities through order invariant confidence is
        strictly higher than if we use naive approach in cases with multiple
        predicted codes (and equal otherwise).
        '''
        probs_order_inv = self.model.predict(
            self.toydata['occ1'], order_invariant_conf=True
        )
        probs_naive = self.model.predict(self.toydata['occ1'], order_invariant_conf=False)

        diff = probs_order_inv['conf'] - probs_naive['conf']
        multiple_predicted_codes = probs_order_inv['hisco_2'] != 'nan'

        # If multiple predicted codes, probabilities should be higher
        self.assertTrue((diff[multiple_predicted_codes] > 0).all())

        # If not multiple codes, probabilities should be identical
        self.assertAlmostEqual(diff[~multiple_predicted_codes].abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
