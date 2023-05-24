
import unittest
import torch
from time import time

from demo.data.sample_coco_test import test as tests
from demo.data.sample_coco_refss import refss

from sacreeos.metric.cider import CiderBase
from sacreeos.metric.cider_r import CiderR
from sacreeos.metric.cider_d import CiderD
from sacreeos.metric.bleu import BLEU

from sacreeos.scst import Scst


class SpeedTest(unittest.TestCase):
    """ Efficiency Test """
    # Note: these tests may fail in case the system does not support OpenMP
    # or multiprocessing

    def test_standalone_metric_speed(self):
        """ Test the fastlib implementation efficiency. """

        def compare_fastlib_speed(self, metric, tests, refss, faster_or_slower, verbose=True):
            t1 = time()
            metric.compute(tests, refss, fast_lib=False)
            no_fast_lib_t = time() - t1

            t1 = time()
            metric.compute(tests, refss, fast_lib=True)
            fast_lib_t = time() - t1

            if verbose:
                print("num cases: " + str(len(tests)) + " - " + str(metric.get_signature()) + " fastlib: " + str(round(fast_lib_t,5)) + " sec, no_fast: "
                      + str(round(no_fast_lib_t, 5)) + " sec, speedup: " + str(no_fast_lib_t/fast_lib_t))
            if faster_or_slower == 'faster':
                self.assertLessEqual(fast_lib_t, no_fast_lib_t)
            elif faster_or_slower == 'slower':
                self.assertLessEqual(no_fast_lib_t, fast_lib_t)
            elif faster_or_slower is None:
                pass

        # without corpus pre initialization
        cider_d = CiderD()
        cider_r = CiderR()
        cider = CiderBase()
        bleu = BLEU()

        # first test to exclude the compilation time
        compare_fastlib_speed(self, cider_d, tests[0:128], refss[0:128], None, False)

        compare_fastlib_speed(self, cider_d, tests[0:128], refss[0:128], 'faster')
        compare_fastlib_speed(self, cider_r, tests[0:128], refss[0:128], 'faster')
        compare_fastlib_speed(self, cider, tests[0:128], refss[0:128], 'faster')
        compare_fastlib_speed(self, bleu, tests[0:128], refss[0:128], 'faster')

        compare_fastlib_speed(self, cider_d, tests[0:2500], refss[0:2500], 'faster')
        compare_fastlib_speed(self, cider_r, tests[0:2500], refss[0:2500], 'faster')
        compare_fastlib_speed(self, cider, tests[0:2500], refss[0:2500], 'faster')
        compare_fastlib_speed(self, bleu, tests[0:2500], refss[0:2500], 'faster')

        compare_fastlib_speed(self, cider_d, tests, refss, 'faster')
        compare_fastlib_speed(self, cider_r, tests, refss, 'faster')
        compare_fastlib_speed(self, cider, tests, refss, 'faster')
        compare_fastlib_speed(self, bleu, tests, refss, 'faster')

        pre_cider_d = CiderD(corpus_refss=refss)
        pre_cider_r = CiderR(corpus_refss=refss)
        pre_cider = CiderBase(corpus_refss=refss)
        # the first invokation involves the slow conversion from Python document frequency dictionary
        # to C std::unordered_map... hence we expect the first calculation to be slower
        compare_fastlib_speed(self, pre_cider_d, tests[0:128], refss[0:128], 'slower')
        compare_fastlib_speed(self, pre_cider_r, tests[0:128], refss[0:128], 'slower')
        compare_fastlib_speed(self, pre_cider, tests[0:128], refss[0:128], 'slower')

        # after the firts time however, the df is now stored in RAM identified by an unique virtual path
        # which the C code can easily access to... hence we expect the calculation to be now faster
        compare_fastlib_speed(self, pre_cider_d, tests[0:128], refss[0:128], 'faster')
        compare_fastlib_speed(self, pre_cider_r, tests[0:128], refss[0:128], 'faster')
        compare_fastlib_speed(self, pre_cider, tests[0:128], refss[0:128], 'faster')
        compare_fastlib_speed(self, pre_cider_d, tests, refss, 'faster')
        compare_fastlib_speed(self, pre_cider_r, tests, refss, 'faster')
        compare_fastlib_speed(self, pre_cider, tests, refss, 'faster')

    def print_loss_calculation_speed(self):
        """ Optionally print the loss calculation efficiency, just for curiosity """

        my_eos_tok = 'EOS'
        my_bos_tok = 'SOS'
        refss_with_eos = []
        for refs in refss:
            refss_with_eos.append([ref + ' ' + my_eos_tok for ref in refs])

        fake_nspi = 3
        base_args = {'nspi': fake_nspi}

        scst_1 = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                      metric_class=Scst.METRIC_CIDER_D,
                      base_class=Scst.BASE_AVERAGE,
                      eos_token=my_eos_tok,
                      corpus_refss=refss_with_eos,
                      base_args=base_args)

        def test_loss_speed(scst, multiplier):
            fake_bs = 2
            fake_max_seq_len = 10

            # simulate logprobs
            torch_preds = -torch.abs(torch.randn(fake_bs * multiplier, fake_nspi, fake_max_seq_len))

            # correct case
            my_preds_1 = [['a dog running in the park EOS', 'a dog is running in a park EOS',
                           'a little dog chasing a ball EOS'],
                          ['a train on the tracks EOS', 'a train is passing through the road EOS',
                           'a train is passing by EOS']] * multiplier
            my_refss_1 = [['a dog chasing in the park EOS', 'a dog is jumping on a park EOS',
                           'a little dog is eating ice cream EOS'],
                          ['a train is going very fast EOS', 'a train passing through cars EOS',
                           'a yellow and blue train EOS']] * multiplier
            t1 = time()
            scst.compute_scst_loss(sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                                   refss=my_refss_1, base_preds=None,
                                   eos_pos_upthresh=fake_max_seq_len,
                                   reduction='mean')
            sacre_t = time() - t1

            cider = CiderD()
            t1 = time()
            flattened_my_preds_1 = [t for my_p in my_preds_1 for t in my_p]
            repeated_my_refss_1 = [refs for refs in my_refss_1 for _ in range(5)]
            t2 = time()
            _, _ = cider.compute(tests=flattened_my_preds_1, refss=repeated_my_refss_1, fast_lib=False)
            cider_t = time() - t1
            cider_ttt = time() - t2

            print("Multiplier: " + str(multiplier) + " Sacre elaps:  " + str(sacre_t)
                  + " s Cider Elaps: " + str(cider_t) + " Cider senza repet: " + str(cider_ttt))

        test_loss_speed(self, scst_1, 1)
        test_loss_speed(self, scst_1, 48)
        test_loss_speed(self, scst_1, 128)
        test_loss_speed(self, scst_1, 256)

