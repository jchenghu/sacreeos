import unittest
from sacreeos.cli import ManualSignature


class ManualSignatureTest(unittest.TestCase):

    def test_manual_signature(self):
        """ Test Scst manual signatures generation. """

        user_inputs = ['', '', '', '', '', 'y', '', '']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'STANDARD_wInit+Cider-D[n4,s6.0]+greedy[nspi5]+1.0.0')

        user_inputs = ['', '', '', '', '', 'n', '', '']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_wInit+Cider-D[n4,s6.0]+greedy[nspi5]+1.0.0')

        user_inputs = ['r', '', '', '', '', '', '', 'n', '', '']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_wInit+Cider-R[n4,rc0.8,lc0.2,a1.0]+greedy[nspi5]+1.0.0')

        user_inputs = ['r', '', '', '', '', 'n', '', 'n', '', '']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_w/oInit+Cider-R[n4,rc0.8,lc0.2,a1.0]+greedy[nspi5]+1.0.0')

        user_inputs = ['r', '', '0.3', '0.7', '', 'n', '', 'n', '', '10']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_w/oInit+Cider-R[n4,rc0.3,lc0.7,a1.0]+greedy[nspi10]+1.0.0')

        user_inputs = ['r', '', '0.3', '0.7', '', 'n', '', 'n', 'm', '10']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_w/oInit+Cider-R[n4,rc0.3,lc0.7,a1.0]+average[nspi10]+1.0.0')

        user_inputs = ['r', '', '0.3', '0.7', '', 'n', '', 'n', 'mEaN', 'M', '10']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_w/oInit+Cider-R[n4,rc0.3,lc0.7,a1.0]+average[nspi10]+1.0.0')

        # try the try catch
        user_inputs = ['r', '1', '', '', '', '2', '0.3', '0.7', '', 'n', '', 'n', 'mEaN', 'M', '10']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_w/oInit+Cider-R[n2,rc0.3,lc0.7,a1.0]+average[nspi10]+1.0.0')

        # trying also previous
        user_inputs = ['b', 'p', 'd', '', '', 'p', 'r', '2', '0.3', '0.7', '', 'n', '', 'n', 'mEaN', 'M', '10']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'NO<EOS>_w/oInit+Cider-R[n2,rc0.3,lc0.7,a1.0]+average[nspi10]+1.0.0')

        user_inputs = ['b', 'n', '', 'y', 'g', 'randomGIbberish', '5']
        self.assertEqual(ManualSignature(interactive=False, inputs_queue=user_inputs,
                                         verbose=False).generate_signature(),
                         'STANDARD_w/oInit+BLEU[n4]+greedy[nspi5]+1.0.0')

