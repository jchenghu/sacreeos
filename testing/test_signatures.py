import unittest
from sacreeos import __VERSION__
from sacreeos.scst import Scst

from demo.data.sample_coco_refss import refss


class SignatureTest(unittest.TestCase):

    def test_signatures(self):
        """ Test Scst signatures according to the arguments. """

        my_eos_tok = '<eos>'
        my_bos_tok = '<sos>'
        refss_with_eos = []
        for refs in refss:
            refss_with_eos.append([ref + ' ' + my_eos_tok for ref in refs])

        scst = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                    metric_class=Scst.METRIC_CIDER_D,
                    base_class=Scst.BASE_GREEDY,
                    eos_token=my_eos_tok,
                    corpus_refss=refss_with_eos)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "STANDARD_wInit+Cider-D[n4,s6.0]+greedy[nspi5]+" + __VERSION__)

        scst = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                    metric_class=Scst.METRIC_CIDER_D,
                    base_class=Scst.BASE_GREEDY,
                    eos_token=my_eos_tok,
                    corpus_refss=None)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "STANDARD_w/oInit+Cider-D[n4,s6.0]+greedy[nspi5]+" + __VERSION__)

        scst = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                    metric_class=Scst.METRIC_CIDER_D,
                    base_class=Scst.BASE_GREEDY,
                    eos_token=my_eos_tok,
                    corpus_refss=refss)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "NO<EOS>_wInit+Cider-D[n4,s6.0]+greedy[nspi5]+" + __VERSION__)

        scst = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                    metric_class=Scst.METRIC_BLEU,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=my_eos_tok,
                    corpus_refss=None)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "NO<EOS>_w/oInit+BLEU[n4]+average[nspi5]+" + __VERSION__)

        scst = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                    metric_class=Scst.METRIC_BLEU,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=my_eos_tok)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "NO<EOS>_w/oInit+BLEU[n4]+average[nspi5]+" + __VERSION__)

        scst = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                    metric_class=Scst.METRIC_CIDER_R,
                    base_class=Scst.BASE_GREEDY,
                    eos_token=my_eos_tok)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "STANDARD_w/oInit+Cider-R[n4,rc0.8,lc0.2,a1.0]+greedy[nspi5]+"
                        + __VERSION__)

        scst = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                    metric_class=Scst.METRIC_CIDER_R,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=my_eos_tok)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "NO<EOS>_w/oInit+Cider-R[n4,rc0.8,lc0.2,a1.0]+average[nspi5]+"
                        + __VERSION__)

        # ------------------------------
        #   metrics arguments testing
        # ------------------------------
        metric_args = {'n': 5, 'repeat_coeff': 0.5, 'length_coeff': 0.5}
        scst = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                    metric_class=Scst.METRIC_CIDER_R,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=my_eos_tok,
                    metric_args=metric_args)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "NO<EOS>_w/oInit+Cider-R[n5,rc0.5,lc0.5,a1.0]+average[nspi5]+"
                        + __VERSION__)

        metric_args = {'n': 5, 'sigma': 0.6}
        scst = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                    metric_class=Scst.METRIC_CIDER_D,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=my_eos_tok,
                    metric_args=metric_args)
        print(scst.get_signature())
        self.assertTrue(scst.get_signature() == "STANDARD_w/oInit+Cider-D[n5,s0.6]+average[nspi5]+"
                        + __VERSION__)

