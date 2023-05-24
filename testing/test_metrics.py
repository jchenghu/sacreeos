import unittest
import numpy as np
from demo.data.sample_coco_test import test as sample_test
from demo.data.sample_coco_refss import refss as sample_refss
from sacreeos.metric.cider import CiderBase
from sacreeos.metric.cider_r import CiderR
from sacreeos.metric.cider_d import CiderD
from sacreeos.metric.bleu import BLEU

EPS = 1e-4


class MetricsTest(unittest.TestCase):

    def test_cider_base(self):
        """ Test cider without initialization. """

        def test_exact_cider(self, cider_d, cider_r, cider, tests, refss):
            cider_d_score_nofast, _ = cider_d.compute(tests, refss, fast_lib=False)
            cider_r_score_nofast, _ = cider_r.compute(tests, refss, fast_lib=False)
            cider_score_nofast, _ = cider.compute(tests, refss, fast_lib=False)
            self.assertTrue(abs(cider_d_score_nofast - 1.3950099700651095) < EPS)
            self.assertTrue(abs(cider_r_score_nofast - 1.4084284297742564) < EPS)
            self.assertTrue(abs(cider_score_nofast - 1.5128477715943824) < EPS)

            cider_d_score_fast, _ = cider_d.compute(tests, refss, fast_lib=True)
            cider_r_score_fast, _ = cider_r.compute(tests, refss, fast_lib=True)
            cider_score_fast, _ = cider.compute(tests, refss, fast_lib=True)
            self.assertTrue(abs(cider_d_score_fast - 1.3950099700651095) < EPS)
            self.assertTrue(abs(cider_r_score_fast - 1.4084284297742564) < EPS)
            self.assertTrue(abs(cider_score_fast - 1.5128477715943824) < EPS)

        def test_compare_cider(self, cider_d, cider_r, cider, tests, refss):
            cider_d_score_nofast, cider_d_array_nofast = cider_d.compute(tests, refss, fast_lib=False)
            cider_r_score_nofast, cider_r_array_nofast = cider_r.compute(tests, refss, fast_lib=False)
            cider_score_nofast, cider_array_nofast = cider.compute(tests, refss, fast_lib=False)

            cider_d_score_fast, cider_d_array_fast = cider_d.compute(tests, refss, fast_lib=True)
            cider_r_score_fast, cider_r_array_fast = cider_r.compute(tests, refss, fast_lib=True)
            cider_score_fast, cider_array_fast = cider.compute(tests, refss, fast_lib=True)

            self.assertTrue(abs(cider_d_score_nofast - cider_d_score_fast) < EPS)
            self.assertTrue(abs(cider_r_score_nofast - cider_r_score_nofast) < EPS)
            self.assertTrue(abs(cider_score_nofast - cider_score_fast) < EPS)
            self.assertTrue(np.all(np.abs(cider_d_array_nofast - cider_d_array_fast) < EPS))
            self.assertTrue(np.all(np.abs(cider_r_array_nofast - cider_r_array_fast) < EPS))
            self.assertTrue(np.all(np.abs(cider_array_nofast - cider_array_fast) < EPS))

        cider_d = CiderD()
        cider_r = CiderR()
        cider = CiderBase()
        test_exact_cider(self, cider_d, cider_r, cider, sample_test, sample_refss)
        test_compare_cider(self, cider_d, cider_r, cider, sample_test[0:1], sample_refss[0:1])
        test_compare_cider(self, cider_d, cider_r, cider, sample_test[0:2500], sample_refss[0:2500])
        test_compare_cider(self, cider_d, cider_r, cider, sample_test, sample_refss)

    def test_preinit_cider(self):
        """ Test pre-initializated cider. """

        def test_equal_complete_res(self, cider, cider_pre, tests, refss):
            pre_cider_score_nofast, pre_cider_array_nofast = cider_pre.compute(tests, refss, fast_lib=False)
            pre_cider_score_fast, pre_cider_array_fast = cider_pre.compute(tests, refss, fast_lib=True)
            cider_score, cider_array = cider.compute(tests, refss, fast_lib=False)
            self.assertTrue(abs(cider_score - pre_cider_score_fast) < EPS)
            self.assertTrue(abs(cider_score - pre_cider_score_nofast) < EPS)
            self.assertTrue(np.all(np.abs(cider_array - pre_cider_array_fast) < EPS))
            self.assertTrue(np.all(np.abs(cider_array - pre_cider_array_nofast) < EPS))

        def test_inequal_partial_res(self, cider, cider_pre, tests, refss):
            pre_cider_score_nofast, pre_cider_array_nofast = cider_pre.compute(tests, refss, fast_lib=False)
            pre_cider_score_fast, pre_cider_array_fast = cider_pre.compute(tests, refss, fast_lib=True)
            cider_score, cider_array = cider.compute(tests, refss, fast_lib=False)
            self.assertTrue(abs(cider_score - pre_cider_score_fast) > EPS)
            self.assertTrue(abs(cider_score - pre_cider_score_nofast) > EPS)

        cider_d = CiderD()
        cider_r = CiderR()
        cider = CiderBase()
        pre_cider_d = CiderD(corpus_refss=sample_refss)
        pre_cider_r = CiderR(corpus_refss=sample_refss)
        pre_cider = CiderBase(corpus_refss=sample_refss)

        test_equal_complete_res(self, cider_d, pre_cider_d, sample_test, sample_refss)
        test_equal_complete_res(self, cider_r, pre_cider_r, sample_test, sample_refss)
        test_equal_complete_res(self, cider, pre_cider, sample_test, sample_refss)

        # AND AGAIN! Since in case of precomputed mode, the first iteration involves
        # different operations compared to the following ones
        test_equal_complete_res(self, cider_d, pre_cider_d, sample_test, sample_refss)
        test_equal_complete_res(self, cider_r, pre_cider_r, sample_test, sample_refss)
        test_equal_complete_res(self, cider, pre_cider, sample_test, sample_refss)

        test_inequal_partial_res(self, cider_d, pre_cider_d, sample_test[1:2], sample_refss[1:2])
        test_inequal_partial_res(self, cider_d, pre_cider_d, sample_test[:2500], sample_refss[:2500])
        test_inequal_partial_res(self, cider_d, pre_cider_d, sample_test[-2500:], sample_refss[-2500:])

        test_inequal_partial_res(self, cider_r, pre_cider_r, sample_test[1:2], sample_refss[1:2])
        test_inequal_partial_res(self, cider_r, pre_cider_r, sample_test[:2500], sample_refss[:2500])
        test_inequal_partial_res(self, cider_r, pre_cider_r, sample_test[-2500:], sample_refss[-2500:])

        test_inequal_partial_res(self, cider, pre_cider, sample_test[1:2], sample_refss[1:2])
        test_inequal_partial_res(self, cider, pre_cider, sample_test[:2500], sample_refss[:2500])
        test_inequal_partial_res(self, cider, pre_cider, sample_test[-2500:], sample_refss[-2500:])

    def test_bleu(self):
        """ Test BLEU scorer. """

        def test_exact_bleu(self, bleu, test, refs):
            c_corpus_bleu, _ = bleu.compute(test, refs, fast_lib=True)
            self.assertTrue(abs(c_corpus_bleu[0] - 0.8236204013377755) < EPS)
            self.assertTrue(abs(c_corpus_bleu[1] - 0.6757971161167308) < EPS)
            self.assertTrue(abs(c_corpus_bleu[2] - 0.5306837684964745) < EPS)
            self.assertTrue(abs(c_corpus_bleu[3] - 0.40722948475820187) < EPS)

            py_corpus_bleu, _ = bleu.compute(test, refs, fast_lib=False)
            self.assertTrue(abs(py_corpus_bleu[0] - 0.8236204013377755) < EPS)
            self.assertTrue(abs(py_corpus_bleu[1] - 0.6757971161167308) < EPS)
            self.assertTrue(abs(py_corpus_bleu[2] - 0.5306837684964745) < EPS)
            self.assertTrue(abs(py_corpus_bleu[3] - 0.40722948475820187) < EPS)

        def test_compare_bleu(self, bleu, test, refs):
            c_corpus_bleu, c_sent_bleu = bleu.compute(test, refs, fast_lib=True)
            py_corpus_bleu, py_sent_bleu = bleu.compute(test, refs, fast_lib=False)
            self.assertTrue(abs(c_corpus_bleu[0] - py_corpus_bleu[0]) < EPS)
            self.assertTrue(abs(c_corpus_bleu[1] - py_corpus_bleu[1]) < EPS)
            self.assertTrue(abs(c_corpus_bleu[2] - py_corpus_bleu[2]) < EPS)
            self.assertTrue(abs(c_corpus_bleu[3] - py_corpus_bleu[3]) < EPS)
            c_sent_bleu = np.array(c_sent_bleu)
            py_sent_bleu = np.array(py_sent_bleu)
            self.assertTrue(np.all(np.abs(c_sent_bleu - py_sent_bleu) < EPS))

        bleu = BLEU()
        test_exact_bleu(self, bleu, sample_test, sample_refss)
        test_compare_bleu(self, bleu, sample_test[0:1], sample_refss[0:1])
        test_compare_bleu(self, bleu, sample_test[0:2500], sample_refss[0:2500])
        test_compare_bleu(self, bleu, sample_test, sample_refss)

    def test_cider_arguments(self):
        """ Test cider using different metrics arguments. """

        def check_inequality(self, cider_1, cider_2, tests, refss):
            cider_1_score_py, _ = cider_1.compute(tests, refss, fast_lib=False)
            cider_2_score_py, _ = cider_2.compute(tests, refss, fast_lib=False)
            cider_1_score_c, _ = cider_1.compute(tests, refss, fast_lib=True)
            cider_2_score_c, _ = cider_2.compute(tests, refss, fast_lib=True)

            print("py_1: " + str(cider_1_score_py) + " c_1:" + str(cider_1_score_c))
            print("py_2: " + str(cider_2_score_py) + " c_2:" + str(cider_2_score_c))

            self.assertFalse(abs(cider_1_score_py - cider_2_score_py) < EPS)
            self.assertFalse(abs(cider_1_score_c - cider_2_score_c) < EPS)
            self.assertTrue(abs(cider_1_score_py - cider_1_score_c) < EPS)
            self.assertTrue(abs(cider_2_score_py - cider_2_score_c) < EPS)

        cider_d_1 = CiderD(sigma=2.0)
        cider_d_2 = CiderD(sigma=6.0)
        check_inequality(self, cider_d_1, cider_d_2, sample_test, sample_refss)

        cider_r_1 = CiderR(repeat_coeff=0.4, length_coeff=0.6)
        cider_r_2 = CiderR(repeat_coeff=0.2, length_coeff=0.8)
        check_inequality(self, cider_r_1, cider_r_2, sample_test, sample_refss)

        cider_r_1 = CiderR(alpha=2.0)
        cider_r_2 = CiderR(alpha=0.5)
        check_inequality(self, cider_r_1, cider_r_2, sample_test, sample_refss)

        print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        cider_d_1 = CiderD(n=5, sigma=2.0)
        cider_d_2 = CiderD(n=6, sigma=6.0)
        check_inequality(self, cider_d_1, cider_d_2, sample_test, sample_refss)

        cider_r_1 = CiderR(n=5, repeat_coeff=0.4, length_coeff=0.6)
        cider_r_2 = CiderR(n=3, repeat_coeff=0.2, length_coeff=0.8)
        check_inequality(self, cider_r_1, cider_r_2, sample_test, sample_refss)

